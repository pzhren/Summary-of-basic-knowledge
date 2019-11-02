[TOC]

# 对文件的操作

## 文件解压缩

### 压缩：

```
sudo tar zcvf work.tar.gz work/
sudo tar jcvf work.tar.bz2 work/
sudo tar cvf work.tar work
```

记住了，要加sudo

### 解压缩：

```
tar xzvf work.tar.gz
tar xjvf work.tar.bz2
tar xvf work.tar
```

------------------------------------------------
版权声明：本文为CSDN博主「努力不脱发选手」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_41147129/article/details/89437134



## 删除文件/文件夹

删除文件夹

```shell
sudo rm -rf /home/ren/.conda/envs/pytorch031/lib/python3.6/h5py
```

## 复制文件

```shell
sudo cp -rfv /home/lelou/anaconda3/lib/python3.6/site-packages/h5py /home/ren/.conda/envs/pytorch031/lib/python3.6
```

- sudo：给予权限
- cp：copy
- r：复制文件夹
- f：强制复制，不用询问
- v：显示复制详情

## 查看文件夹大小

```python
(base) lpx@lpx-Super-Server:~$ cd /home/lpx/bottom-up-attention1
(base) lpx@lpx-Super-Server:~/bottom-up-attention1$ du -sh
1.4G	.
```

## 修改文件（夹）的权限

```python
sudo chmod -R 777 renpengzhen/bottom-up-attention1/
```

给予最高权限

# 目录问题

``` shell
./ 当前目录。
../ 父级目录。
/ 根目录。
```

# 安装包问题

```shell
anaconda search -t conda tensorflow #搜索关于可安装的tensorflow包,找到合适的版本
conda search opencv-python
anaconda show jjhelmus/tensorflow #显示安装的详细信息
```
# 大文件上传至github

在该大文件所在的目录下打开`Git Bash Here`,运行下面代码，后面是大文件的名字。

`git lfs track "model.ckpt.data-00000-of-00001"`