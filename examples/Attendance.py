# 采用face_recognition包检测人脸去识别,考勤，即通过数据库的照片训练k近邻分类器，然后测试未知样本。
#################################
# 1-引用模块
#################################
import math
from sklearn import neighbors
import  os
import os.path
import pickle
from PIL import Image,ImageDraw
import face_recognition           #安装见https://blog.csdn.net/fb_941219/article/details/89415300
import face_recognition as fr
import sys #用于调试时，使用。

from face_recognition.face_recognition_cli import image_files_in_folder #文件夹中的图像
from sklearn.neighbors import KNeighborsClassifier  #导入k近邻分类器
import sys

#################################
# 2-训练模型
#################################
#1.建立一个数据集x:128维度加上y一个维度，总共是129维度。
# 2.对每一个照片操作
# 3.决定n
# 4.训练出分类器
# 5.保存分类器

def train(train_dir,model_save_path='trained_knn_model.clf',n_neighbors=3,knn_algo='ball_tree'):
    '''
    功能：训练一个KNN分类器
    :param train_dir: 训练目录。其下对每个已知的人，分别以其名字，建立一个文件夹
    :param model_save_path:模型保存的位置
    :param n_neighbors:邻居数默认为3
    :param knn_algo: 支持KNN的数据结构；ball_tree是一种树型结构。
    :return: KNN分类器

    '''
    #生成训练集
    x=[] #注意x最终是18维。
    y=[]
    #遍历训练集中的每一个人
    for class_dir in os.listdir(train_dir):  #http://www.runoob.com/python/os-listdir.html
        if not os.path.isdir(os.path.join(train_dir,class_dir)): #判断是否是目录
            continue #结束当前循环，进入下一个循环。
        #遍历这个人的每一张照片
        for image_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image=fr.load_image_file(image_path) #传入人脸的位置
            boxes=fr.face_locations(image)  #定出人脸位置
            #对于当前图片，增加编码到训练集合
            x.append(fr.face_encodings(image,known_face_locations=boxes)[0])  #编码返回时一个128维度的向量。
            y.append(class_dir)
    #决定n
    if n_neighbors is None:
        n_neighbors=3
    #训练出分类器
    knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(x,y)

    #保存分类器
    if model_save_path is not None:
        with open(model_save_path,'wb') as f:#wb，b二进制
            pickle.dump(knn_clf,f)   #模型存储
    return knn_clf

# 预测
# prediction=predict(full_file_path,model_path='trained_knn_model.clf')
def predict(x_img_path,knn_clf=None,model_path=None,distance_threshold=0.35):
    '''
    利用KNN分类器识别给定的照片中的人脸
    :param x_imag_path:必须对应照片的地址而不是照片的文件夹
    :param knn_clf:
    :param distance_threshold:
    :return: [(人名1，边界盒子1),...]
    '''
    if knn_clf is None and model_path is None:
        raise Exception('必须提供KNN分类器：可选方式为knn_clf或model_path')
    #加载训练好的KNN模型（如果有）
    #rb表示要读入的二进制数据
    print('调试断点0')
    # sys.exit(0)
    if knn_clf is None:
        print(model_path) #trained_knn_model.clf
        with open(model_path,'rb') as f:          #https://www.cnblogs.com/tianyiliang/p/8192703.html

            knn_clf=pickle.load(f)
            print(f)
            print('调试断点1','*'*20)
            # sys.exit(1)
    #加载图片，发现人脸的位置
    print(x_img_path)
    # sys.exit(2)
    x_img=fr.load_image_file(x_img_path)  #https://blog.csdn.net/MG_ApinG/article/details/82252954
    # x_img=fr.load_image_file(x_img_path)  #Permission denied没有权限，原因是load_image_file需要打开文件的地址而不是文件夹的地址。
    print('调试断点2','*'*20)
    # sys.exit(2)
    x_face_location=fr.face_locations(x_img)
    print(x_face_location)
    print('调试断点3','*'*20)
    # sys.exit(3)

    #对测试图片中的人脸进行编码
    encodings=fr.face_encodings(x_img) #http://www.360doc.com/content/18/0403/18/48868863_742603302.shtml
    print(encodings)
    print(len(encodings[0])) #128个数据组成的向量
    print('调试断点4','*'*20)
    # sys.exit(4)
    x_face_locations=fr.face_locations(x_img)
    print('调试断点5','*'*20)
    print(x_face_locations)
    # sys.exit(5)

    #利用KNN模型，找出与测试人员最匹配的人脸
    #encodings:128个人脸特征的向量
    closest_distace=knn_clf.kneighbors(encodings,n_neighbors=3)
    print('调试断点6','*'*20)
    print(closest_distace) #(array([[0.34381298, 0.35287966, 0.35839984]]), array([[3, 2, 7]], dtype=int64))
    # sys.exit(6)

    are_matches=[closest_distace[0][i][0]<=distance_threshold  for i in range(len(x_face_locations))] #匿名函数， are_matches即是否匹配上
    print('调试断点7','*'*20)
    print(are_matches)
    # sys.exit(7)

    #预测类别，并remove classifications that aren't within the threshold即移除不在阀值内的分类
    print(knn_clf.predict(encodings))
    print(list(x_face_locations))
    print(list(are_matches))
    print(list(zip(knn_clf.predict(encodings),x_face_locations,are_matches)))
    #pred 预测值，loc头像位置
    return [(pred,loc) if rec else ('unknown',loc) for pred,loc,rec in zip(knn_clf.predict(encodings),x_face_locations,are_matches)]  #zip 压缩http://www.runoob.com/python/python-func-zip.html

#结果可视化
def show_names_on_image(img_path,predictions):
    '''
    人脸识别可视化
    :param img_path: 待识别图片的位置
    :param predictions: 预测的结果
    :return:
    '''
    pil_image=Image.open(img_path).convert('RGB') #将图片转换成格式  #https://blog.csdn.net/icamera0/article/details/50843172
    draw=ImageDraw.Draw(pil_image)                #ImageDraw类支持各种几何图形的绘制和文本的绘制https://blog.csdn.net/guduruyu/article/details/71213717
    for name,(top,right,bottom,left) in predictions:
        #用pillow模块画图人脸边界盒子
        draw.rectangle( ((left,top),(right,bottom)),outline=(255,0,255))

        #pillow 里可能生成UTF-8格式，所以这里做如下转换
        #这里有draw不能解码出name字体的问题。
        name=name.encode('utf-8')
        name=name.decode('ISO-8859-1')
        print('要打印的name是',type(name))
        #在人脸下写下名字，作为标签
        # sys.exit(1)
        text_width,text_height=draw.textsize(name)

        draw.rectangle(((left,bottom-text_height-10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
        draw.text((left,bottom-text_height-10),name,(255,0,255))
        #遍加名字到li_names
        li_names.append(name)

    #从内存删除draw
    del draw
    #显示结果图
    pil_image.show()

#######################
#统计分析
#######################
#为了打印名字的集合
li_names=[]
#计算总人数
def count(train_dir):
    '''
    counts the total number of the set
    :param train_dir:
    :return:
    '''
    path=train_dir
    count=0
    for fn in os.listdir(path): #fn表示的是文件名
        count=count+1
    return count
#获取所有名字的列表
def list_all(train_dir):
    '''
    determine the list of all names
    :param train_dir:
    :return:
    '''
    path=train_dir
    result=[]
    for fn in os.listdir(path):#fn表示的是文件夹名
        result.append(fn)
    return result
#输出结果
def stat_output():
    s_list=set(li_names)
    s_list_all=set(list_all('examples/train'))
    if 'unknown'in s_list:
        s_list.remove('unknown')
    print('查阅',s_list)
    tot_num=count('examples/train')
    s_absent=set(s_list_all-s_list) #未到人数
    print('\n')
    print('***********************\n')
    print('全体名单',s_list_all)
    print('已到名单',s_list)
    print('应到人数',tot_num)
    print('已到人数',len(s_list))
    print('出勤率:{:.2f}'.format(float(len(s_list))/float(tot_num)))
    print('未到：',s_absent)

#######################
#运行
#######################
if __name__ == '__main__': #主函数
    # sys.exit()
    #1. 训练KNN分类器（它可以保存，以便再使用）

    #如果有模型时，就不要进行训练了。

    # print('正在训练KNN分类器')
    # classifier=train('examples/train',model_save_path='trained_knn_model.clf',n_neighbors=3)
    # print('完成费时间的KNN分类器训练，训练结束')

    # 2  利用训练好的分类器，对新图片进行预测
    for image_file in os.listdir('examples/test'): #对测试文件夹
        for picture_flie in os.listdir('examples/test/{}'.format(image_file)):
            full_file_path=os.path.join('examples/test/{}'.format(image_file),picture_flie)  #图片对应的完整的位置，image_file对应图片对应的文件夹
            print( 'full_file_path：',full_file_path)
            print(picture_flie)
            # sys.exit(0)
            print('正在在 {} 中寻找人脸ing'.format(image_file))
            # sys.exit(0)
            # 利用分类器，找出所有的人脸
            # 要么传递一个classifier文件名，要么一个classifer模型实例
            prediction=predict(full_file_path,model_path='trained_knn_model.clf')
            print(prediction)
            print('完成{}一次{}人脸预测'.format(image_file,picture_flie),'+'*50)
            # sys.exit(1)

        #打印结果

            for name,(top,right,bottom,left) in prediction:
                print('发现人脸 :{}；   人脸位置：（{}，{}，{}，{}）'.format(name,top,right,bottom,left))
            # sys.exit()

            # # 在图片上显示预测结果
            show_names_on_image(os.path.join('examples/test/{}'.format(image_file),picture_flie),prediction)
    # sys.exit()
    #3.输出统计结果
    stat_output()
