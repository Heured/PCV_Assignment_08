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

 
    
