# AI Study

* [Colabolatory](https://research.google.com/colaboratory)
* [Kaggle](https://www.kaggle.com)
\
\
__My Github Repositories__
* [Datascience](https://github.com/dev-onejun/Datascience)
* [Statistical Artificial Intelligence](https://github.com/dev-onejun/Statistical-Artificial-Intelligence)
* [Lablup Conference 2022](https://github.com/dev-onejun/Lablup-Conference-2022)

## Theoretical References

* A type of Neural Networks and the concepts
  * https://untitledtblog.tistory.com/154

## Practical References

1. How to install tensorflow

  * The [LINK](https://www.tensorflow.org/install/docker) goes to tensorflow docs how to install it on docker.
  * [LINK](https://hub.docker.com/r/tensorflow/tensorflow) goes to tensorflow page of docker hub.
    * it does not work on docker too, so i should compile tensorflow manually.
      * [1] https://www.tensorflow.org/install/source
      * [2] http://www.kwangsiklee.com/2017/04/텐서플로우-경고메세지-해결하기-the-tensorflow-library-wasnt-compiled-to-use-sse3-instructions/
      * you can just see [1] document only.

  * Acclerate to use GPU in M1 Silicon
    * https://developer.apple.com/metal/tensorflow-plugin/

  * How to set jupyter-notebook password and https connection
    * https://jupyter-notebook.readthedocs.io/en/stable/public_server.html

  * After using i5-10400F and gtx 1660ti,
    * I can use docker images pushed on docker hub by tensorflow/tensorflow.
    * But building on docker didn't work properly, which means it couldn't detect gpus, I might seem that the reason is the different version of cuda and cuDNN which are installed on my host mahcine and docker container. I installed the cuda version, 11.7 and the container has the cuda version, 11.2.
    * So I think it will be fixed when I build sources locally, but because I can use docker images now, I have not tried yet.
    * OR I experienced that when updating packages in container, tensorflow doesn't work properly which emits the same errors above. so before building sources on container, making packages update will be the same result with build locally. (It is a speculation, still ..)
    * Now I'm using .whl file built from container and install it to container, gpu-jupyter

  * I wrote articles for this subject in my blog. [Go](https://dev-onejun.github.io/study/artificial%20intelligence/2022/12/15/InstallLibrary.html)

2. Kubeflow

  * https://www.kangwoo.kr/2020/03/04/kubeflow-%EC%86%8C%EA%B0%9C/

3. jupyter-notebook

  * [This](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) is docs about docker jupyter images. After reading this, you can select docker images what you should use.
