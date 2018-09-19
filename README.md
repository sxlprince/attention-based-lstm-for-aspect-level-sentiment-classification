代码取自：https://github.com/scaufengyang/TD-LSTM       
论文解读：https://zhuanlan.zhihu.com/p/42659009   

注：这里实现的是aspect-term嵌入，而不是aspect嵌入。  
    target是句子中直接存在的名词或实体,是aspect-term；aspect指的是名词或实体类别，即aspect-category。  
    例如：Staffs are not that fridedlly,but the taste covers all.  
    其中Staffs是target， 对应aspect-term，service是aspect，对应aspect-category.
