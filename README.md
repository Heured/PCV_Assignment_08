# PCV_Assignment_08
Image search

## 图像搜索
  
### 矢量空间模型
  矢量空间模型是一个用于表示和搜索文本文档的模型。该名字来源于用矢量来表示文本文档，这些矢量是由文本词频直方图构成的。矢量包含了每个单词出现的次数，而且在其他别的地方包含很多0元素，由于其忽略了单词出现的顺序及位置，该模型也被称为BOW表示模型。
  
  最常用的权重是tf(词频)，表示本单词在本文中出现的次数除以本文中单词的总数。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/CiPin.jpg)  
  
  
  idf(逆向文档频率):  
  ![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/NiXiangwdpl.jpg)  
  其中|D|是在语料库D中文档的数目，分母是语料库中包含单词w的文档数d。将两者相乘可以得到矢量v中对应元素的tf-idf权重。  
  
### 视觉单词
  为了将文本挖掘技术应用到图像中，首先需要建立视觉等效单词。通常可以采用SIFT局部描述子做到。它的思想是将描述子空间量化成一些典型实例，并将图像中的每个描述子指派到其中某个实例中。这些典型实例可以通过分析训练图像集来确定，即视觉单词。所有这些视觉单词的集合成为视觉词汇(视觉码本)。
  
  从一个训练图像集提取特征描述子，利用聚类算法可以构建出视觉单词。聚类算法中最常用的K-means。用视觉单词直方图来表示图像，则该模型便称为BOW模型。
  
### 过程
  vocabulary类中包含了一个由单词聚类中心VOC与每个单词对应的逆向文档频率构成的向量。  
  在索引图像前，需要建立一个数据库。这里，对图像的索引就是从这些图像中提取描述子，利用词汇将描述子转换成视觉单词，并保存视觉单词及对应图像的单词直方图。从而可以利用数据库进行查询，并返回相似的图像作为搜索结果。  
  从数据库中搜索图像过程中，我们用BoW(bag-of-word,词袋模型)来表示整个图像，不过这里介绍的过程是通用的，可以应用于寻找相似的物体、相似的脸、相似的颜色等，完全取决于图像及所用的描述子。

  
## 步骤
1.生成代码所需模型文件

  
  
```python
# -*- coding: utf-8 -*
import pickle
from PCV.imagesearch import vocabulary
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift

##要记得将PCV放置在对应的路径下

# 获取图像列表
imlist = get_imlist('./data/') ###要记得改成自己的路径
nbr_images = len(imlist)

# 获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# 提取文件夹下图像的sift特征
for i in range(nbr_images):
    sift.process_image(imlist[i], featlist[i])

# 生成词汇
voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist, 1000, 10)
#保存词汇
# saving vocabulary
with open('./data/vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)

print('vocabulary is:', voc.name, voc.nbr_words)
```

2.将模型数据导入数据库

  
  
```python
# -*- coding: utf-8 -*
import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

##要记得将PCV放置在对应的路径下
##要记得将PCV放置在对应的路径下

# 获取图像列表
imlist = get_imlist('./data/')##记得改成自己的路径
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# load vocabulary
#载入词汇
with open('./data/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
#创建索引
indx = imagesearch.Indexer('testImaAdd.db',voc)
indx.create_tables()
# go through all images, project features on vocabulary and insert
#遍历所有的图像，并将它们的特征投影到词汇上
for i in range(nbr_images)[:1000]:
    locs,descr = sift.read_features_from_file(featlist[i])
    indx.add_to_index(imlist[i],descr)
# commit to database
#提交到数据库
indx.db_commit()

con = sqlite.connect('testImaAdd.db')
print(con.execute('select count (filename) from imlist').fetchone())
print(con.execute('select * from imlist').fetchone())
```

3.测试

  
  
```python
# -*- coding: utf-8 -*
import pickle
from PCV.localdescriptors import sift
from PCV.imagesearch import imagesearch
from PCV.geometry import homography
from PCV.tools.imtools import get_imlist

##要记得将PCV放置在对应的路径下
##要记得将PCV放置在对应的路径下

# load image list and vocabulary
#载入图像列表
imlist = get_imlist('./data/')##要改成自己的地址
nbr_images = len(imlist)

#载入特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# 载入词汇
with open('./data/vocabulary.pkl', 'rb') as f:     ##要改成自己的地址
    voc = pickle.load(f)

src = imagesearch.Searcher('testImaAdd.db', voc)

# index of query image and number of results to return
#查询图像索引和查询返回的图像数
q_ind = 0
nbr_results = 20

# regular query
# 常规查询(按欧式距离对结果排序)
res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
print('top matches (regular):', res_reg)

# load image features for query image
#载入查询图像特征
q_locs,q_descr = sift.read_features_from_file(featlist[q_ind])
fp = homography.make_homog(q_locs[:,:2].T)

# RANSAC model for homography fitting
#用单应性进行拟合建立RANSAC模型
model = homography.RansacModel()
rank = {}

# load image features for result
#载入候选图像的特征
for ndx in res_reg[1:]:
    locs,descr = sift.read_features_from_file(featlist[ndx])  # because 'ndx' is a rowid of the DB that starts at 1
# get matches
    matches = sift.match(q_descr, descr)
    ind = matches.nonzero()[0]
    ind2 = matches[ind]
    tp = homography.make_homog(locs[:,:2].T)
    # compute homography, count inliers. if not enough matches return empty list
    try:
        H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)
    except:
        inliers = []
    # store inlier count
    rank[ndx] = len(inliers)

# sort dictionary to get the most inliers first
sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
res_geom = [res_reg[0]]+[s[0] for s in sorted_rank]
print('top matches (homography):', res_geom)

# 显示查询结果
imagesearch.plot_results(src,res_reg[:8]) #常规查询
imagesearch.plot_results(src,res_geom[:8]) #重排后的结果
```
  
  
结果：  
  
![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/3_03.PNG)  

![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/3_01.png)  

![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/3_02.png)  

4.建立演示程序以及web应用

  
  
```python
# -*- coding: utf-8 -*
import cherrypy
import pickle
import urllib
import os
from numpy import *
#from PCV.tools.imtools import get_imlist
from PCV.imagesearch import imagesearch
import random

""" This is the image search demo in Section 7.6. """


class SearchDemo:

    def __init__(self):
        # 载入图像列表
        self.path = './data/'
        #self.path = 'D:/python_web/isoutu/first500/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        #self.imlist = get_imlist('./first500/')
        #self.imlist = get_imlist('E:/python/isoutu/first500/')
        self.nbr_images = len(self.imlist)
        print(self.imlist)
        print(self.nbr_images)
        #print(str(len(self.imlist))+"###############")
        #self.ndx = range(self.nbr_images)
        self.ndx = list(range(self.nbr_images))
        print(self.ndx)

        # 载入词汇
        # f = open('first1000/vocabulary.pkl', 'rb')
        with open('./data/vocabulary.pkl','rb') as f:
            self.voc = pickle.load(f)
        #f.close()

        # 显示搜索返回的图像数
        self.maxres = 10

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """

    def index(self, query=None):
        self.src = imagesearch.Searcher('testImaAdd.db', self.voc)

        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:

            # query the database and get top images
            # 查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"

                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print(imname+"################")
                html += "</a>"
            # show random selection if no query
            # 如果没有查询图像则随机显示一些图像
        else:
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"

                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print(imname+"################")
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True

# conf_path = os.path.dirname(os.path.abspath(__file__))
#conf_path = os.path.join(conf_path, "service.conf")
#cherrypy.config.update(conf_path) #cherrypy.quickstart(SearchDemo())

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
```
  
  
结果：  
  
![emmmm](https://github.com/Heured/PCV_Assignment_08/blob/master/ImgToShow/5_01.PNG)  

### 遇到问题
  
  1.第三个文件test.py
  ```python
  Traceback (most recent call last):
  File "D:/pyCharm/pycharm_workspace/2019_4_30_ShiJue/test.py", line 32, in <module>
    res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\PCV\imagesearch\imagesearch.py", line 128, in query
    h = self.get_imhistogram(imname)
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\PCV\imagesearch\imagesearch.py", line 96, in get_imhistogram
    return pickle.loads(str(s[0]))
TypeError: a bytes-like object is required, not 'str'
  ```
  解决办法：
  ```python
  # import中
  import operator
  from functools import cmp_to_key
  
  #方法修改为
  def get_imhistogram(self,imname):
        """ Return the word histogram for an image. """
        
        im_id = self.con.execute(
            "select rowid from imlist where filename='%s'" % imname).fetchone()
        s = self.con.execute(
            "select histogram from imhistograms where rowid='%d'" % im_id).fetchone()
        
        # use pickle to decode NumPy arrays from string
        # 从return pickle.loads(str(s[0]))修改为：
        return pickle.loads(s[0])
   def candidates_from_histogram(self,imwords):
        """ Get list of images with similar words. """
        
        # get the word ids
        words = imwords.nonzero()[0]
        
        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates+=c
        
        # take all unique words and reverse sort on occurrence 
        tmp = [(w,candidates.count(w)) for w in set(candidates)]
        # 从tmp.sort(cmp=lambda x,y:cmp(x[1],y[1]))修改为：
        tmp.sort(key=cmp_to_key(lambda x,y:operator.gt(x[1],y[1])))
        tmp.reverse()
        
        # return sorted list, best matches first    
        return [w[0] for w in tmp] 
  ```
 2.第四个文件show.py
 ```python
 [08/May/2019:18:29:56] ENGINE Listening for SIGTERM.
[08/May/2019:18:29:56] ENGINE Bus STARTING
[08/May/2019:18:29:56] ENGINE Set handler for console events.
[08/May/2019:18:29:56] ENGINE Started monitor thread 'Autoreloader'.
[08/May/2019:18:29:56] ENGINE Serving on http://127.0.0.1:8080
[08/May/2019:18:29:56] ENGINE Bus STARTED
[08/May/2019:18:30:00] HTTP 
Traceback (most recent call last):
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\cherrypy\_cprequest.py", line 628, in respond
    self._do_respond(path_info)
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\cherrypy\_cprequest.py", line 687, in _do_respond
    response.body = self.handler()
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\cherrypy\lib\encoding.py", line 219, in __call__
    self.body = self.oldhandler(*args, **kwargs)
  File "D:\Anaconda\Anaconda3_5.3.0\envs\no6py3\lib\site-packages\cherrypy\_cpdispatch.py", line 54, in __call__
    return self.callable(*self.args, **self.kwargs)
  File "D:/pyCharm/pycharm_workspace/2019_4_30_ShiJue/show.py", line 72, in index
    random.shuffle(self.ndx)
  File "mtrand.pyx", line 4859, in mtrand.RandomState.shuffle
  File "mtrand.pyx", line 4862, in mtrand.RandomState.shuffle
TypeError: 'range' object does not support item assignment
 ```
  
  原因：是python3中range不返回数组对象，而是返回range对象,加个声明为list的语句就行
  
  
 解决办法：  
 ```python
     def __init__(self):
        # 载入图像列表
        self.path = './data/'
        #self.path = 'D:/python_web/isoutu/first500/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        #self.imlist = get_imlist('./first500/')
        #self.imlist = get_imlist('E:/python/isoutu/first500/')
        self.nbr_images = len(self.imlist)
        print(self.imlist)
        print(self.nbr_images)
        #print(str(len(self.imlist))+"###############")
        # 从self.ndx = range(self.nbr_images)修改为：
        self.ndx = list(range(self.nbr_images))
        print(self.ndx)
 ```
