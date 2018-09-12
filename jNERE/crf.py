#
# From https://github.com/rguthrie3/BiLSTM-CRF/blob/master/model.py
#

import dynet
import numpy as np


class CRF():

    def __init__(self, model, id_to_tag):

        self.id_to_tag = id_to_tag
        self.tag_to_id = {tag: id for id, tag in id_to_tag.items()}
        self.n_tags = len(self.id_to_tag)
        self.b_id = len(self.tag_to_id)
        self.e_id = len(self.tag_to_id) + 1

        self.transitions = model.add_lookup_parameters((self.n_tags+2,
                                                 self.n_tags+2),
                                                name="transitions")

    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dynet.scalarInput(0)
        tags = [self.b_id] + tags
        for i, obs in enumerate(observations):
            # print self.b_id
            # print self.e_id
            # print obs.value()
            # print tags
            # print self.transitions
            # print self.transitions[tags[i+1]].value()
            score = score \
                    + dynet.pick(self.transitions[tags[i + 1]], tags[i])\
                    + dynet.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dynet.pick(self.transitions[self.e_id], tags[-1])
        return score


    def viterbi_loss(self, observations, tags):
        observations = [dynet.concatenate([obs, dynet.inputVector([-1e10, -1e10])], d=0) for obs in
                        observations]
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        if viterbi_tags != tags:
            gold_score = self.score_sentence(observations, tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dynet.scalarInput(0), viterbi_tags


    def neg_log_loss(self, observations, tags):
        observations = [dynet.concatenate([obs, dynet.inputVector([-1e10, -1e10])], d=0) for obs in observations]
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score


    def forward(self, observations):
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dynet.pick(scores, argmax_score)
            max_score_expr_broadcast = dynet.concatenate([max_score_expr] * (self.n_tags+2))
            return max_score_expr + dynet.log(
                dynet.sum_dim(dynet.transpose(dynet.exp(scores - max_score_expr_broadcast)), [1]))

        init_alphas = [-1e10] * (self.n_tags + 2)
        init_alphas[self.b_id] = 0
        for_expr = dynet.inputVector(init_alphas)
        for idx, obs in enumerate(observations):
            # print "obs: ", obs.value()
            alphas_t = []
            for next_tag in range(self.n_tags+2):
                obs_broadcast = dynet.concatenate([dynet.pick(obs, next_tag)] * (self.n_tags + 2))
                # print "for_expr: ", for_expr.value()
                # print "transitions next_tag: ", self.transitions[next_tag].value()
                # print "obs_broadcast: ", obs_broadcast.value()

                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dynet.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.e_id]
        alpha = log_sum_exp(terminal_expr)
        return alpha


    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars = [-1e10] * (self.n_tags + 2)
        init_vvars[self.b_id] = 0  # <Start> has all the probability
        for_expr = dynet.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.n_tags + 2)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.n_tags + 2):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dynet.pick(next_tag_expr, best_tag_id))
            for_expr = dynet.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.e_id]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dynet.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.b_id
        # Return best path and best path's score
        return best_path, path_score