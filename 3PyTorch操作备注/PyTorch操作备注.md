# 普通图片读取

## 使用from PIL import Image 读取图片

```python
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

%matplotlib inline

img_path = "dog.jpg"
image = Image.open(img_path)#type(image)：PIL.JpegImagePlugin.JpegImageFile
img_torch = transforms.ToTensor()(image) #把一个取值范围是[0,255]的PIL.Image 转换成 Tensor，torch.Size([3, 1200, 1200])
plt.imshow(img_torch.numpy().transpose(1,2,0)) #数据类型tensor-->numpy转换，(1200, 1200, 3)
plt.show()
```

![1563328834464](PyTorch操作备注.assets/1563328834464.png)

## 使用import matplotlib.image as mpimg 读取图片

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片

# 读取图片
dog = mpimg.imread('timg.jpg')
plt.subplot(121)
plt.imshow(dog)
plt.title('Original Image')
#形状reshape，测试是否可以在变换后进行还原到原来的图像上
dog = dog.reshape(1,3,1200,1200)
dog = dog.reshape(1200,1200,3)
plt.subplot(122)
plt.imshow(dog) #(1200,1200,3)
plt.title('Original-->Reshape Image')
plt.show()
```

![1563329092861](PyTorch操作备注.assets/1563329092861.png)

# shape和size的区别

## [numpy] 

### .size：计算数组和矩阵所有数据的个数 

```python
a = np.array([[1,2,3],[4,5,6]]) 
np.size(a)#返回值为 6 
np.size(a,1)#返回值为 3
a.size ##返回值为 6
```

### .shape():得到矩阵每维的大小 

```python
a = np.array([[1,2,3],[4,5,6]]) 
np.shape(a) #返回值为 (2,3)
a.shape #返回值为 (2,3)
```

>另外要注意的是，shape和size既可以作为函数，也可以作为ndarray的属性 

## [pytorch]

### .shape

```python
theta = torch.tensor([
    [1, 0, -0.2],
    [0, 1, -0.4]
], dtype=torch.float)
theta.shape #torch.Size([2, 3])
```
### .size()
```python
theta = torch.tensor([
    [1, 0, -0.2],
    [0, 1, -0.4]
], dtype=torch.float)
theta.size() #torch.Size([2, 3])
```



# pytorch中squeeze()和unsqueeze()函数介绍

## **unsqueeze()函数**：增加维度



![img](PyTorch操作备注.assets/20180812155855509.png)

可以看出a的维度为（2，3）

在第二维增加一个维度，使其维度变为（2，1，3）

![img](PyTorch操作备注.assets/20180812160119403.png)

可以看出a的维度已经变为（2，1，3）了，同样如果需要在倒数第二个维度上增加一个维度，那么使用b.unsqueeze(-2)

## **squeeze()函数介绍**：减小维度

1.首先得到一个维度为（1，2，3）的tensor（张量）

![img](PyTorch操作备注.assets/20180812160833709.png)

由图中可以看出c的维度为（1，2，3）

2.下面使用squeeze()函数将第一维去掉

![img](PyTorch操作备注.assets/20180812161010282.png)

可见，维度已经变为（2，3）

3.另外

![img](PyTorch操作备注.assets/20180812161246184.png)

可以看出维度并没有变化，仍然为（1，2，3），这是因为只有维度为1时才会去掉。