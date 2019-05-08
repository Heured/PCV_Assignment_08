# PCV_Assignment_08
3d modeling


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
