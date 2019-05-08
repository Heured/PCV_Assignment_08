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
        tmp.sort(key=cmp_to_key(lambda x,y:operator.gt(x[1],y[1])))
        tmp.reverse()
        
        # return sorted list, best matches first    
        return [w[0] for w in tmp] 
  ```
 
    
