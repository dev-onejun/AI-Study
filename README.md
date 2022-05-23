# AI Study
* [Colabolatory](https://research.google.com/colaboratory)
* [Kaggle](https://www.kaggle.com)
## References
1) How to install tensorflow
    - It is recommended that using docker, because tensorflow uses specific CPU instruction(ex. AVX)
        and etc ...
    - The [LINK](https://www.tensorflow.org/install/docker) goes to tensorflow docs how to install it on docker.
    - [LINK](https://hub.docker.com/r/tensorflow/tensorflow) goes to tensorflow page of docker hub.
    - UPDATE!!!
        - it does not work on docker too, so i should compile tensorflow manually.
            - [1] https://www.tensorflow.org/install/source
            - [2] http://www.kwangsiklee.com/2017/04/텐서플로우-경고메세지-해결하기-the-tensorflow-library-wasnt-compiled-to-use-sse3-instructions/
            - you can just see [1] document only.

    * Acclerate to use GPU in M1 Silicon
        - https://developer.apple.com/metal/tensorflow-plugin/

    * How to set jupyter-notebook password and https connection
        - https://jupyter-notebook.readthedocs.io/en/stable/public_server.html

2) Kubeflow
    - https://www.kangwoo.kr/2020/03/04/kubeflow-%EC%86%8C%EA%B0%9C/
