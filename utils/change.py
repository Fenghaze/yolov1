# -- coding: utf-8 --
# @Time : 2020/8/22 14:16
# @Author : zhl
# @File : change.py
# @Desc: 

path = "../../data/coins/images.txt"

f = open(path, "r", encoding="utf-8")

content = f.read()
content1 = content.replace(',', ' ')

ff = open('./images.txt', "w")
ff.write(content1)
ff.flush()