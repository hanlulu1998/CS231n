在本作业中，你将实现循环网络，并将其应用于在微软的COCO数据库上进行图像标注。我们还会介绍TinyImageNet数据集，然后在这个数据集使用一个预训练的模型来查看图像梯度的不同应用。本作业的目标如下：

- 理解*循环神经网络（RNN）*的结构，知道它们是如何随时间共享权重来对序列进行操作的。
- 理解普通循环神经网络和长短基记忆（Long-Short Term Memory）循环神经网络之间的差异。
- 理解在测试时如何从RNN生成序列。
- 理解如何将卷积神经网络和循环神经网络结合在一起来实现图像标注。
- 理解一个训练过的卷积神经网络是如何用来从输入图像中计算梯度的。
- 进行高效的交叉验证并为神经网络结构找到最好的超参数。
- 实现图像梯度的不同应用，比如显著图，搞笑图像，类别可视化，特征反演和DeepDream。

## 安装

有两种方法来完成作业：在本地使用自己的机器，或者使用[http://Terminal.com](https://link.zhihu.com/?target=http%3A//Terminal.com)的虚拟机。

### 云端作业

Terminal公司为我们的课程创建了一个单独的子域名：[www.stanfordterminalcloud.com](https://link.zhihu.com/?target=https%3A//www.stanfordterminalcloud.com/)。在该域名下注册。作业2的快照可以在[这里](https://link.zhihu.com/?target=https%3A//www.stanfordterminalcloud.com/snapshot/49f5a1ea15dc424aec19155b3398784d57c55045435315ce4f8b96b62819ef65)找到。如果你注册到了本课程，就可以联系上助教（更多信息请上Piazza）来得到用来做作业的点数。一旦你启动了快照，所有的环境都是为你配置好的，马上就可以开始作业。我们在Terminal上写了一个简明[教程](https://link.zhihu.com/?target=http%3A//cs231n.github.io/terminal-tutorial/)。

### 本地作业

点击[此处](https://link.zhihu.com/?target=http%3A//cs231n.stanford.edu/winter1516_assignment3.zip)下载代码压缩文件。初次之外还有些库间依赖的配置：

**[选项1]使用Anaconda**：推荐方法是安装[Anaconda](https://link.zhihu.com/?target=https%3A//www.continuum.io/downloads)，它是Python的一个发布版，包含了最流行的科研、数学、工程和数据分析Python包。一旦安装了它，下面的提示就都可略过，准备直接开始写作业吧。*译者注：推荐。*

**[选项2]手动安装，虚拟环境**：如果你不想用Anaconda，想要走一个充满风险的手动安装路径，那么可能就要为项目创建一个[虚拟环境](https://link.zhihu.com/?target=http%3A//docs.python-guide.org/en/latest/dev/virtualenvs/)了。如果你不想用虚拟环境，那么你的确保所有代码需要的依赖关系都是景在你的机器上被安装了。要建立虚拟环境，运行下面代码：

```text
cd assignment3
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

**下载数据**：一旦得到作业初始代码，你就需要下载CIFAR-10数据集，然后在assignment1目录下运行下面代码：*译者注：也可手动下载解压后放到**cs231n/datasets目录**。*

```text
cd cs231n/datasets 
./get_coco_captioning.sh
./get_tiny_imagenet_a.sh
./get_pretrained_model.sh
```

**编译Cython扩展包**：卷积神经网络需要一个高效的实现。我们使用[Cython](https://link.zhihu.com/?target=http%3A//cython.org/)实现了一些函数。在运行代码前，你需要编译Cython扩展包。在cs231n目录下，运行下面命令：

```text
python setup.py build_ext --inplace
```

**启用IPython**：得到了CIFAR-10数据集之后，你应该在作业assignment1目录中启用IPython notebook的服务器，如果对IPython notebook不熟悉，可以阅读[教程](https://link.zhihu.com/?target=http%3A//cs231n.github.io/ipython-tutorial)。

**注意**：如果你是在OSX上的虚拟环境中工作，可能会遇到一个由matplotlib导致的错误，原因在[这里](https://link.zhihu.com/?target=http%3A//matplotlib.org/faq/virtualenv_faq.html)。你可以通过在assignment2目录中运行start_ipython_osx.sh脚本来解决问题。

## 提交作业

无论你是在云终端还是在本地完成作业，一旦完成作业，就运行collectSubmission.sh脚本；这样将会产生一个assignment3.zip的文件，然后将这个文件上传到你的dropbox中这门课的[作业页面](https://link.zhihu.com/?target=https%3A//coursework.stanford.edu/portal/site/W15-CS-231N-01/)。

### Q1：使用普通RNN进行图像标注（40分）

IPython Notebook文件**RNN_Captioning.ipynb**将会带你使用普通RNN实现一个在微软COCO数据集上的图像标注系统。

### Q2：使用LSTM进行图像标注（35分）

IPython Notebook文件**LSTM_Captioning.ipynb**将会带你实现LSTM，并应用于在微软COCO数据集上进行图像标注。

### Q3：图像梯度：显著图和高效图像（10分）

IPython Notebook文件**NetworkVisualization-PyTorch.ipynb（NetworkVisualization-TensorFlow.ipynb）**将会介绍TinyImageNet数据集。你将使用一个训练好的模型在这个数据集上计算梯度，然后将其用于生成显著图、类可视化、愚弄图像等等。

### Q4：图像生成：类别，反演和风格迁移（30分）

在IPython Notebook文件**StyleTransfer-PyTorch**中，你将会实现风格迁移，把一幅图片的风格转移到另一幅图片上。

### Q5：做点儿其他的！（+10分） 

根据作业内容，做点够酷的事儿。比如作业中没有讲过的其他生成图像的方式？

## Q6：GANs（编者后添加）

在IPython Notebook文件**GANs-PyTorch**中生成对抗网络，用minst检验你的成果。



