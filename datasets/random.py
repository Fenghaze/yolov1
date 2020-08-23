# -- coding: utf-8 --
# @Time : 2020/8/22 18:33
# @Author : zhl
# @File : random.py
# @Desc: 打乱原始数据集，按8:2划分数据集

from random import shuffle


def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


result=[]
with open('images.txt','r') as f:
	for line in f:
		result.append(list(line.strip('\n').split(',')))
shuffle(result)

text_save("val.txt", result[:3310])
text_save("train.txt", result[3310:])