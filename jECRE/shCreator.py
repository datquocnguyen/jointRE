# -*- coding: utf-8 -*-

import os
import sys
import re

os.chdir("../")
sys.setrecursionlimit(500000)
sys.path.append(os.path.abspath(""))
os.chdir("./jECRE")


def shTrainingCreator(EXPpath, PreTrainedVectorsPath):
    pretrainedVectors = {}
    for path1, subdirs, files in os.walk(PreTrainedVectorsPath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        for name in files:
            if name.endswith(".vectors.xz"):
                langCode = name[:name.find(".vectors.xz")]
                pretrainedVectors[langCode] = path + name
                print(langCode, path + name)

    prefix = "#!/bin/bash\n#PBS -q normalbw\n#PBS -l mem=10G\n#PBS -P sa4\n#PBS -l walltime=47:55:00\n#PBS -l wd\n#PBS -l ncpus=1\n\n"
    prefix += "module load gcc/4.9.0\nmodule load python/2.7.13\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8\n\n"
    prefix += "source /short/kl1/dqn576/.DyNet/bin/activate\n\n"

    prefix += "cd /short/kl1/dqn576/jPTDP\n"

    count = 0

    for path1, subdirs, files in os.walk(EXPpath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        dev = None
        train = None
        for name in files:
            if name.endswith("dev.conllu") > 0:
                dev = name
            if name.endswith("train.conllu") > 0:
                train = name

        # print path, train, dev

        if train is not None and dev is not None:
            count += 1
            writer = open("scripts/" + str(count) + ".sh", "w")
            writer.write(prefix)
            writer.write(
                "python jPTDP.py --dynet-seed 123456789 --dynet-mem 5000 --epochs 30 --lstmdims 128 --lstmlayers 2 --wembedding 100 --cembedding 50 --pembedding 100 --model model --params model.params --outdir " + path + " --train " + path + train + " --dev " + path + dev)

            langCode = None
            if train.startswith("no_bokmaal"):
                langCode = "no_bokmaal"
            elif train.startswith("no_nynorsk"):
                langCode = "no_nynorsk"
            else:
                if train.find("_") > 0:
                    langCode = train[:train.find("_")]
                else:
                    langCode = train[:train.find("-")]

            pretrained = None
            if langCode in pretrainedVectors:
                pretrained = pretrainedVectors[langCode]
                # print(pretrained)

            print(langCode, path + train, pretrained)

            if pretrained is not None:
                writer.write(" --prevectors " + pretrained + "\n")

            writer.write("\n\n")
            writer.close()


def shTestingCreator(EXPpath, prefix):
    count = 0
    writer = open("scripts/" + prefix + ".sh", "w")
    for path1, subdirs, files in os.walk(EXPpath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        dev = None
        train = None
        test = None
        for name in files:
            if name.endswith("dev.conllu") > 0:
                dev = name
            if name.endswith("train.conllu") > 0:
                train = name
            if name.endswith("test.conllu") > 0:
                test = name

        print path, train, dev

        if train is not None and test is not None:
            count += 1

            writer.write("#!/bin/bash")
            writer.write("\n#PBS -q normal")
            writer.write("\n#PBS -l mem=7G")
            writer.write("\n#PBS -P kl1")
            writer.write("\n#PBS -l walltime=47:00:00")
            writer.write("\n#PBS -l wd")
            writer.write("\n#PBS -l ncpus=1")
            writer.write(
                "\nmodule load gcc/4.9.0\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8")

            writer.write(
                "\ncd /home/dqnguyen/workspace/DependencyParsing/jPTDP \n")

            writer.write(
                "python jPTDP.py --dynet-seed 123456789 --predict --model " + path + "model --params " + path + "model.params --outdir " + path + " --test " + path + test + " --output " + test + ".pred \n")

            writer.write("\n")
    writer.close()


def shTestingCreator_dev(EXPpath, prefix):
    count = 0
    writer = open("scripts/" + prefix + ".sh", "w")
    for path1, subdirs, files in os.walk(EXPpath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        dev = None
        train = None
        test = None
        for name in files:
            if name.endswith("dev.conllu") > 0:
                dev = name
            if name.endswith("train.conllu") > 0:
                train = name
            if name.endswith("test.conllu") > 0:
                test = name

        print path, train, dev

        if train is not None and dev is not None:
            count += 1

            writer.write("#!/bin/bash")
            writer.write("\n#PBS -q normal")
            writer.write("\n#PBS -l mem=3G")
            writer.write("\n#PBS -P kl1")
            writer.write("\n#PBS -l walltime=47:00:00")
            writer.write("\n#PBS -l wd")
            writer.write("\n#PBS -l ncpus=1")
            writer.write(
                "\nmodule load gcc/4.9.0\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8")

            writer.write(
                "\ncd /home/dqnguyen/workspace/DependencyParsing/jPTDP \n")

            writer.write(
                "python jPTDP.py --predict --dynet-seed 123456789 --model " + path + "model --params " + path + "model.params --outdir " + path + " --test " + path + dev + " --output " + dev + ".pred \n")

            writer.write("\n")
    writer.close()


def shTrainingCreator_Rocket(EXPpath, PreTrainedVectorsPath):
    pretrainedVectors = {}
    for path1, subdirs, files in os.walk(PreTrainedVectorsPath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        for name in files:
            if name.endswith(".vectors.xz"):
                langCode = name[:name.find(".vectors.xz")]
                pretrainedVectors[langCode] = path + name
                print(langCode, path + name)

    count = 0

    for path1, subdirs, files in os.walk(EXPpath):
        subdirs.sort(reverse=False)
        path = path1.replace("\\", "/") + "/"
        dev = None
        train = None
        for name in files:
            if name.endswith("dev.conllu") > 0:
                dev = name
            if name.endswith("train.conllu") > 0:
                train = name

        # print path, train, dev

        if train is not None and dev is not None:
            count += 1
            writer = open("scripts/" + str(count) + ".sh", "w")

            # prefix = "#!/bin/bash\n#PBS -q normalbw\n#PBS -l mem=10G\n#PBS -P sa4\n#PBS -l walltime=47:55:00\n#PBS -l wd\n#PBS -l ncpus=1\n\n"
            # prefix += "module load gcc/4.9.0\nmodule load python/2.7.13\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8\n\n"
            # prefix += "source /short/kl1/dqn576/.DyNet/bin/activate\n\n"
            # prefix += "cd /short/kl1/dqn576/jPTDP\n"

            prefix = "#!/bin/bash\n#SBATCH -A mtnihrio\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem-per-cpu=10G\n#SBATCH -p long\n#SBATCH -o " + str(
                count) + ".sh.o.%j\n#SBATCH -e " + str(count) + ".sh.e.%j\n#SBATCH --export=ALL\n"
            prefix += "\nmodule load Python/2.7.13-intel-2017.03-GCC-6.3\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8\n\n"
            prefix += "\nsource /mnt/nfs/home/ntv7/.DyNet/bin/activate\n"
            prefix += "\ncd /mnt/nfs/home/ntv7/CoNLL2018/jPTDP\n"

            writer.write(prefix)
            writer.write(
                "python jPTDP.py --dynet-seed 123456789 --dynet-mem 3000 --hidden2 100 --epochs 30 --lstmdims 128 --lstmlayers 2 --wembedding 100 --cembedding 50 --pembedding 100 --model model_NoPreTrained --params model_NoPreTrained.params --outdir " + path + " --train " + path + train + " --dev " + path + dev)

            langCode = None
            if train.startswith("no_bokmaal"):
                langCode = "no_bokmaal"
            elif train.startswith("no_nynorsk"):
                langCode = "no_nynorsk"
            else:
                if train.find("_") > 0:
                    langCode = train[:train.find("_")]
                else:
                    langCode = train[:train.find("-")]

            pretrained = None
            if langCode in pretrainedVectors:
                pretrained = pretrainedVectors[langCode]
                # print(pretrained)

            print(count, langCode, path + train, pretrained)

            if pretrained is not None:
                writer.write(" --prevectors " + pretrained + "\n")

            writer.write("\n\n")
            writer.close()


def shTrainingjNERE_Creator_Rocket():
    count = 0
    for wordembed in ['50', '100']:
        for charembed in ['25', '50']:
            for nerembed in ['50', '100']:
                for posembed in ['50', '100']:
                    for lr in ['0.001', '0.0005']:
                        for lstmsize in ['100']:

                            count += 1

                            writer = open("scripts/" + str(count) + ".sh", "w")

                            # prefix = "#!/bin/bash\n#PBS -q normalbw\n#PBS -l mem=10G\n#PBS -P sa4\n#PBS -l walltime=47:55:00\n#PBS -l wd\n#PBS -l ncpus=1\n\n"
                            # prefix += "module load gcc/4.9.0\nmodule load python/2.7.13\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8\n\n"
                            # prefix += "source /short/kl1/dqn576/.DyNet/bin/activate\n\n"
                            # prefix += "cd /short/kl1/dqn576/jPTDP\n"

                            prefix = "#!/bin/bash\n#SBATCH -A mtnihrio\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem-per-cpu=4G\n#SBATCH -p long\n#SBATCH -o " + str(
                                count) + ".sh.o.%j\n#SBATCH -e " + str(count) + ".sh.e.%j\n#SBATCH --export=ALL\n"
                            prefix += "\nmodule load Python/2.7.13-intel-2017.03-GCC-6.3\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\nexport LANGUAGE=en_US.UTF-8\n\n"
                            prefix += "\nsource /mnt/nfs/home/ntv7/.DyNet/bin/activate\n"
                            prefix += "\ncd /mnt/nfs/home/ntv7/jECRE\n"

                            prefix += "python jECRE.py --dynet-seed 123456789 --dynet-mem 512 --epochs 50 --wembedding {} --cembedding {} --nembedding {} --pembedding {} --lr {} --lstmdims {}".format(wordembed, charembed, nerembed, posembed, lr, lstmsize)
                            writer.write(prefix)



#                            if wordembed == '50':
#                                writer.write(" --prevectors ../vecs.lc.over100freq.txt.gz")

 #                           if wordembed == '100':
 #                               writer.write(" --prevectors ../glove.6B.100d.txt")

                            writer.write(" --output outputs/exp" + str(count) + "_")

                            writer.write("\n\n")
                            writer.close()


if __name__ == "__main__":
    shTrainingjNERE_Creator_Rocket()
    # shTrainingCreator("/short/kl1/dqn576/CoNLL2017/conll2017-small-datasets", "128-64-small-run", 128, 64)
    # shTestingCreator("/home/dqnguyen/workspace/DependencyParsing/jPTDP-models-full/64-32-128/conll2017-small-datasets", "64-32-test")
    # shTestingCreator("/home/dqnguyen/workspace/DependencyParsing/ud-treebanks-v1.3", "ud-1.3-test")
    # shTestingCreator_dev("/home/dqnguyen/workspace/DependencyParsing/jPTDP-models-conll2017", "rerun-conll-dev")
    # shTestingCreator_dev("/home/dqnguyen/workspace/DependencyParsing/Pre-trained-jPTDP-models", "rerun-pretrained-dev")


