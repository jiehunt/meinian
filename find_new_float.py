
# coding: utf-8

# In[32]:
import time
import pandas as pd
import gc
import numpy as np
import os
from contextlib import contextmanager
import re
import pandas as pd
import numpy as np
data=pd.read_csv('float_clean_data.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
test_lenth=9538
test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = test['vid']
target=train.columns[2:7]

cols = list(set(train.columns)- set(target))
vid=data['vid']
temp=data[cols].select_dtypes(include=['object'])
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
# with timer ("loading ..."):
# #     test = pd.read_csv('test_041717.csv')
# #     train = pd.read_csv('train_041717.csv')

# temp = pd.concat([train, test])
cols = temp.columns
mapping ={
         '详见纸质报告'   : np.nan,
         '详见报告'      : np.nan,
         '详见报告。'     : np.nan,
         '详见报告单'    : np.nan,
         '详见图文报告'  : np.nan,
         '详见报告单。'  : np.nan,
         '见纸质报告单'   : np.nan,
         '见报告单'      :np.nan,
         '见图文报告'    :np.nan,
         '详见纸质报告单' :np.nan,
         '详见纸质报告 详见纸质报告' : np.nan,
         '详见报告单 详见报告单' : np.nan,
         '详见检验单'    : np.nan,
         '未查'         : np.nan,
         '弃查'          :np.nan,
         '未做'          : np.nan,
        '未触及'            :np.nan,
        '未触及 未触及'      :np.nan,
         '标本已退检'     : np.nan,
         '降脂后复查'     : np.nan, # 191
        '具体内容请见分析报告。':np.nan,
        '详见医生诊疗。'      :np.nan,
        '详见中医养生报告'   :np.nan,


        '已完成，建议复诊，咨询电话：52190566转588':1,

         '-'             :0,
         '--'            :0,
         '- -'           :0,
         '-----'         :np.nan,
         '---'           :np.nan,
         '----'          :np.nan,
         '阴性'          :0,
         ' 阴性'         :0,
         '阴性（-）'     :0,
         '阴性(-)'       :0,
         '阴性 阴性'      :0,
         '阴性 -'         :0,
         '- 阴性'         :0,
         '阴性（-） 阴性'  :0,

         '+'             :2,
         '阳性'          :2,
         '阳性(+)'       :2,
         '阳性（+）'     :2,
         '≧1:80阳性'     :2,
          '陽性'         :2,
         '弱阳性'        :1,
         '弱阳'          :1,  #  300036
         '弱阳性(±)'     :1,  #  300036
         '++'           :3,
         '+++'          :4,
         '++++'         :5,
         '＋'           :2,
         '＋＋'          :3,
         '＋＋＋'        :4,
         '阳性(轻度)'    :2,
         '阳性(重度)'    :4,
         '阳性(中度)'    :3,

         '+-'           :2,
         '阴阳'         :2,
         '阴性 +'       :2,
         '不定'         :3,

         '少见'          :1,
         '多见'          :2,

         '正常'          :0,
         '正常 正常'      :0,
         '未见异常'      :0,
         '未见明显异常'   :0,
         '未发现异常'     :0,
         '自述不查'       :0,
         '未发现异常, 未发现异常' :0,
         '未检'           :np.nan,
         '未见异常 未见异常' :0,
         '未见 未见'      :0,
         '全心功能未发现明显异常' :0,
         '耳鼻喉检查未见异常' :0,
         '未见异'        :0,
         '未检出'        :0,
         '未见异常 未见异常 未见异常':0,
         '正常 正常 正常 正常' :0,
         '正常 正常 正常 正常 正常':0,
         '正常 正常 正常' :0,
         '异常'          :2,

         '无 无'         :0,
         '未见'          :0,
         '无'            :0,

         '敏感(S)'       :1,
         '耐药(R)'       :2,
         '耐药'          :2,
         '中度敏感(MS)'  :3,
         '敏感'          :4,

         'Ⅰ'            :1,
         'Ⅱ'            :2,
         'Ⅲ'            :3,
         'Ⅳ'            :4,

         '1+'            :1,
         '+1'            :1,
         '3+'            :3,
         '2+'            :2,

         '无敏感菌'       :0,
         '中介'           :1,
         '中敏'           :2,

         '5级'            :5,
         '4级'            :4,
         '3级'            :3,
         '2级'            :2,
         '1级'            :1,
         '0级'            :0,

         '未检测到缺失。'   :0,
         '未检测到缺失'     :0,
         '基因缺失'        :1,
         '未检测到突变。'   :0,
         '基因突变，IVS-II-654位点突变基因杂合子。' :2,

         '听力下降'        :1,
         '下降'            :1,
         '右耳听力下降可能' :1,

         'HIV抗体阴性'     :0,
         'HIV抗体阴性(-)'   :0,
         'HIV感染待确定'    :1,
         '待复查'           :1,

         'yellow'          :1,
         '黄色'            :1,
          '淡黄色'         :2,
          '浅黄色'         :2,
         '褐色'            :3,
         '黄褐色'          :4,
         '深黄色'          :4,
         '黄棕色'          :5,
          '白浊'           :6,
         '黑色'            :7,
         '淡红色'          :8,
         '暗红色'          :9,
          '红色'           :8,
         '无色'            :10,
         '其他'            :11,

         '透明'            :0,
         '混浊'           :1,
         '浑浊'            :1,

         '软'              :1,
         '软,糊状'          :2,
         '半稀便'          :3,
         '稀'              :4,
          '中'             :5,
          '硬'             :6,

         'O'               :1,
         'O型'             :1,
         '“O”型'           :1,
         '“O”'             :1,
         'O 型'            :1,
         'O  型'           :1,
         '0'               :1,
         '0型'             :1,
         'O型血'           :1,
         'A型'             :2,
          'a型'            :2,
         'A'               :2,
         '(A)'             :2,
          '“A”型'          :2,
          '“A”'            :2,
          'A  型'          :2,
          'B型'            :3,
          'B'              :3,
        '“B”型'            :3,
         'B  型'           :3,
           'B 型'          :3,
         'b型'             :3,
         'AB'              :4,
         'AB型'            :4,
         'AB 型'           :4,
         '“AB”型'          :4,
          '(AB)'           :4,
          '“AB”'           :4,

         '无神经定位体征'    :1,
         '右上下肢肌力减弱'  :2,
         '左肢肌力减弱'      :3,
         '无神经定位体征 无神经定位体征' :4,

         '无压痛点'         :0,
         '叩击痛'           :1,
         '压痛'             :2,
         '叩击痛, 叩击痛'    :1,
         '压痛, 叩击痛'      :3,

         '女性肿瘤指标'     : -1, # 300076

         '4.03 4.03'       :4.03,
         '2.1.'            :2.1,
         '9.871 9.87'      :9.871,
         '36.0 36.0'       :36.0,
        '2..99'            :2.99,
         '44.7 44.7'       :44.7,
         '阴性1.496'       :1.496,
        '346.45 346.45'    :346.45,
        '.45.21'         : 45.21,
         '1.308 1.308'    : 1.308,
         '7.01 11.04'      :11.04,
         '4.50 4.50'       :4.50,
          '0.47 0.47'      :0.47,
         '41.64 41.64'     :41.64,
         '3.85 3.85'       :3.85,
         '59.80 59.80'     :59.80,
         '32.10 32.10'     :32.10,
         '77..21'          :77.21,
         '3。89'           :3.89,
         '16.7.07'         :16.7,
         '5..0'            :5.0,
         '.45.21'          :45.21,
         '16.2-'           :16.2,
         '4.42 4.42'       :4.42,
         '32..5'           :32.5,
         '99 99'           :99,
         '98%'             :98,
         '98 98'           :98,
        'nan 96'           :96,
         '99 nan'          :99,
         '26.2 nan'        :26.2,
         '1.00 1.00'       :1.0,
         'nan nan'         :np.nan,
         '6s'              :6,
         '3.0.0'           :3.0,
         '5.4 10'          :5.41,
         '43.5 43.0'       :43.5,
         '1389 nan'        :1389,
         '0.85 0.65'       :0.85,
         '1.01 0.40'       :1.01,
         '0.00-25.00'      :12.5,
         '0.01（阳性）'     :0.01,
         '/'               :np.nan,
         '3.4~33.9'        :3.4,
         '2.6 2.9'         :2.6,
         '﹥250.00'        :250,
         '5.32 4.39'       :5.32,
         '9.70 11.41'      :9.7,
         '78.02 89.02'     :78.02,
          '1.84 1.93'      :1.84,
         '10.3 14.6'       :10.3,
         '6.93 9.67'       :6.93,
         '0':0,
        '0':0,
        '0-1':1,
        '视不见':0,
        '顶壁': 1,
        '/'  : np.nan,
        '未查 1':np.nan,
        '10^4':6,
        'S':5,
        '4':4,
        '2-5':4,
        '上皮细胞少量':1,
        '粘液丝+' :2   ,
        '无神经定位体征 无神经定位体征':0,
        '无神经定位体征':0,
        '生理反射存在，病理反射未引出':1,
        '双侧腮导口充血':1,
        '疑似左侧良性肿大':1,
        '未生长':1,
        '详见纸质报告单或查询电子版（http://www.xyzjkjt.com/）':np.nan,
        '生理年龄报告查询请登录电子版网址（http://www.xyzjkjt.com/）账号：美年体检号           密码：出生年月日  例（19880810）':np.nan,
        '宫颈刮片：':np.nan,
        '结果详见中医经络检测报告':np.nan,
        '未提示':np.nan,
        '未生长':0,
        '中医体质辨识未发现明显异常':0,
        '少许':1,
        '1.0 1.0':1,
        '慢性喉炎':1,
        '软便':1,
        '＜100':80,
        '轻度的糖尿病或糖尿病前期风险（糖基化正产物偏高）':1,
        '糖尿病或糖尿病前期风险较小':1,
        '少':1,
        '1:100' :1,
        '1:320 +':3,
        '1:100 +':1,
        '见报告':np.nan,
        '巴氏I级,未见异常':0,
        '送检玻片一张，镜检：正常范围内，未见癌细胞。':1,
        '送检玻片一张，镜检：上皮细胞量较少，建议复查。':1,
        '送检玻片一张，镜检：上皮细胞量较少，建议复查。':1,
        '抽烟，不喝酒':1,
        '混合性红细胞':1,
        '未提示':0,
        '非均一性':2,
        '未及':0,
        '脾脏切除':1,
        '视不见':0,
        '有，建议进一步专科检查':1,
        '少量':0,
        '中量':1,
        '慢性喉炎':1,
        '可见粘液丝':1,
        '（尿液）送检浅黄色液体约40ml，TCT制片。镜检：极少量鳞状上皮及尿路上皮细胞，未见癌细胞。':0,
        '（尿液）送检黄色液体约40ml，离心沉淀TCT制片，巴氏染色。镜下：见少量的尿路上皮细胞，未找到癌细胞。':0,
        '（尿液）送检黄色液体约40ml，离心沉淀TCT制片一张，巴氏染色。镜下：见少量的尿路上皮细胞，未找到癌细胞。':0,
        '（尿液）送检黄色液体约45ml，TCT制片。镜检：中量鳞状上皮细胞及少量尿路上皮细胞，未见癌细胞。':0,
        'exit':0,
        'I':1,
        '萎缩性舌炎':1,
        '沟纹舌':2,
        '+/HP':np.nan,
        '＜20':16,
        '0':0,
        '3-5':4,
        '查见':np.nan,
        '十二指肠穿孔修补术术后':1,
         '右眼：角膜0.6mm，前房2.7mm，晶状体3.8mm，玻璃体14.9mm，眼轴24mm；左眼：角膜0.7mm，前房2.6mm，晶状体3.5mm，玻璃体14.6mm，眼轴23mm；':1,
        '0-3':2,
        'R':3,
        '#NAME?':np.nan,
        '<1.00E+03':1,
        '0-2':1,
    '#NUM!':np.nan,
    '2+/HP':2,
'11月1日':np.nan,
    '光感':np.nan,
    '指数':np.nan,
'微弱阳性':np.nan,
'#NUM!'    :np.nan,
'0-5':    4,
 '0-4':3,   
'3.05 3.05':3.05,
    
    '0.96 0.78':0.8,
     '1..27':1.27,
     '0.04 0.04' :0.04,  
    '﹢':0,
    '50.4 50.4':50.4,
 '79.6 79.6':79.6,
    '心内各结构未见明显异常':np.nan,
'脂肪肝':80,
 '6.31.0.45'   :6.31,
'未查 1.0':np.nan,
'***.**':np.nan,
'弃查 弃查':np.nan,
'阳性(低水平)':np.nan,
'- +':np.nan,
'＜0.30':0.0,
'可疑'     :np.nan, 
    '**.*' :np.nan, 
'60 弃查':60,
'fo':np.nan,
         }

mapping2 = {
    # 1321
    '0.4 nan nan nan' : 0.4,
    '未查 未查': np.nan,
    'nan nan nan nan': np.nan,
    'nan nan nan': np.nan,
     'nan 1.2 nan nan nan':1.2,
    'nan 1.0':1,
    '1.0 nan':1,
    'nan nan 1.5':1.5,
    '失明' : 0,
    '手动':np.nan,
         '义眼':np.nan,
    '0.1, 1.0':0.1,
    '1.0 1.0 nan':1.0,
    '0.8 0.8 0.8':0.8,
    'nan 0.4 nan':0.4,
    '0.6 nan nan':0.6,
    'nan nan 0.4':0.6,
    '0.25 0.25 0.25':0.25,
     '1.2 1.2 1.2':1.2,
    '1.0 1.0 1.0':1.0,
    '0.6 0.6 0.6':0.6,
    '1.5 1.5 1.5':1.5,
    '无光感':0,
    'nan 0.3 nan':0.3,
    '0.8 nan nan':0.8,
    'nan 1.0 nan':1.0,
    'nan 0.05 nan':0.05,
'nan nan 0.3':0.3,
'未查 1.2':1.2,
'nan nan nan nan 1.0':1.0,
'0.7 0.7 0.7':0.7,
'nan nan nan nan nan':np.nan,
'1.0 nan nan':1.0,
'0.7 0.7 0.7 0.7':0.7,
'0.2 0.2 0.2':0.2,
'1.0 nan nan 1.0 1.0':1.0,
'nan 0.5 nan':0.5,
'未要求检查 ':np.nan,
'nan 0.6 nan':0.6,
'nan nan 0.7':0.7,
'0.6 nan nan nan nan':0.6,
'0.6 0.6 0.6 0.6 0.6':0.6,
'0.8 未查':0.8,
'nan 1.5 nan nan':1.5,
'未查 0.5':0.5,
'0.9 0.9 0.9 0.9':0.9,
'未查 0.3':0.3,
'0.4 nan nan':0.4,
'0.3 nan nan':0.3,
'nan 0.15 nan':1.5,
'nan 0.9 nan':0.9,
'nan 1.5 nan':1.5,
'nan 弃查':np.nan,
'nan nan 0.5':0.5,
'因无法配合不能检测':np.nan,
'nan 0.8 nan':0.8,
'0.7 nan nan':0.7,
'nan nan 0.6':0.6,
'0.9 nan nan':0.9,
'未要求检查':np.nan,
'0.3 0.3 0.3':0.3,
'0.5 0.5 0.5':0.5,
'0.6 0.6 0.6 0.6':0.6,
'1.2 1.2 1.2 1.2':1.2,

'阴性（-） 阴性（-）':1,
    #'424'
    '心率正常':80,
    '窦性心动过缓':56,
    '窦性心动过速':106,
    '次/分' : np.nan,
}


with timer ("mapping ..."):
    temp = temp.applymap(lambda x : mapping[x] if x in mapping.keys() else x )
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('>') else x)
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('<') else x)
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('﹤') else x)
    temp = temp.applymap(lambda x : x[:-1] if str(x).endswith('.') else x)
    temp = temp.applymap(lambda x : mapping2[x] if x in mapping2.keys() else x )

obj_list_4 = []
obj_list = []

unit_mapping = ['kpa', 'db/m', '(ng/mL)', '(pmol/L)', '(U/ml)', '%', '＜','kg','(umol/L)']
def unit_transform_s(x):
    y = x
    for k in unit_mapping:
        if str(x).endswith(k) > 0:
            return str(x).strip(k)
    return y

def unit_transform_x(x):
    y = re.sub(r'^<(.*)', r'\1', str(x))
    return y
def unit_transform_y(x):
    y = re.sub(r'^>(.*)', r'\1', str(x))
    return y
def unit_transform_space(x):
    y = x
    p = re.compile(r'^(\d.*) (\d.*)')
    m =  p.match(x)
    if m:
        try:
            if float(m.group(1)) == float(m.group(2)):
                y = float(m.group(1))
            else :
                y = float(m.group(1)) + float(m.group(2)) /2
        except:
            y = x
    return y

cols = list(set(cols) - set(['vid']))
for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        try:
            temp[col] = temp[col].apply(lambda x : unit_transform_y(x) if str(x).find('>')==0 else x)
        except:
            print (col)

for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        try:
            temp[col] = temp[col].apply(lambda x : unit_transform_x(x) if str(x).find('<')==0 else x)
        except:
            print (col)

for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        try:
            temp[col] = temp[col].apply(lambda x : unit_transform_space(x) if str(x).find(' ')>0 else x)
        except:
            print (col)

for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        try:
            temp[col] = temp[col].apply(lambda x : unit_transform_s(x))
        except:
            print (col)

def clean(x):
    x=str(x)
    try:
        return re.findall(r"\d+\.?\d*", x)[0]     
    except:
        if  '<' in x :
            return float(x[x.index('<')+1:])
        if  '>' in x :
            return float(x[x.index('>')+1:])
        if  '＜' in x :
            return float(x[x.index('＜')+1:])
        if  '＜=' in x :
            return float(x[x.index('＜=')+1:])
        if  '\t' in  x :
            return float(x[x.index('\t')+1:])
        if len(x.split(sep='.'))>2:#2.2.8
            i=x.rindex('.')
            x=x[0:i]+x[i+1:]
            return float(x)
        if '未做' in x or '未查' in x or '弃查' in x:
            return np.nan
        if str(x).isdigit()==False and len(str(x))>4:
            x=x[0:4]
            return float(x)
        else:
            return x
    
for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        try:
            temp[col] = temp[col].apply(lambda x:clean(x))
        except:
            print (col)                 
            
def unit_transform_nan(x):
    y = x
    p = re.compile(r'^(\d.*) (nan)')
    q = re.compile(r'^(nan) (\d.*)')
    m =  p.match(x)
    n =  q.match(x)
    if m:
        try:
            y = float(m.group(1))
        except:
            y = x
    elif n:
        try:
            y = float(n.group(2))
        except:
            y = x
    
    return y

s_cols = ['1321']
for s_col in s_cols:
    temp[s_col] = temp[s_col].apply(lambda x : unit_transform_nan(str(x)) if pd.notnull(x) else x)
    
def unit_transform_yinyang(x):
    y = x
    p = re.compile(r'阳性 (\d.*)')
    q = re.compile(r'阴性 (\d.*)')
    m =  p.match(x)
    n =  q.match(x)
    if m:
        y = float(m.group(1))
    elif n:
        y = float(n.group(1))
    return y

s_cols = ['2233', '2229']
for s_col in s_cols:
    temp[s_col] = temp[s_col].apply(lambda x : unit_transform_yinyang(str(x)) if pd.notnull(x) else x)
    
def unit_transform_xt(x):
    y = x
    p = re.compile(r'.*(\d{2,3})次/分.*')
    m =  p.match(x)
    q = re.compile(r'.*心动过缓(\d{2,3})')
    n =  q.match(x)
    if m:
        try:
            y = float(m.group(1))
        except:
            y = m.group(1)
    elif n:
        try:
            y = float(n.group(1))
        except:
            y = n.group(1)
    return y

def unit_transform_xt2(x):
    y = x
    p = re.compile(r'^(\d{2,3}).*\D$')
    m =  p.match(x)
    if m:
        try:
            y = float(m.group(1))
        except:
            y = m.group(1)
    return y

def unit_transform_xt3(x):
    y = x
    p = re.compile(r'^\D*(\d{2,3}).*$')
    m =  p.match(x)
    if m:
        try:
            y = float(m.group(1))
        except:
            y = m.group(1)
    return y

s_cols = ['424']
for s_col in s_cols:
    temp[s_col] = temp[s_col].apply(lambda x : unit_transform_xt(str(x)) if pd.notnull(x) else x)
    temp[s_col] = temp[s_col].apply(lambda x : unit_transform_xt2(str(x)) if pd.notnull(x) else x)
    temp[s_col] = temp[s_col].apply(lambda x : unit_transform_xt3(str(x)) if pd.notnull(x) else x)


for col in cols:
    if (np.array(temp[col]).dtype) == 'object':
        obj_list.append(col)
        try:
            temp[col] = temp[col].apply(lambda x : float(x) )
        except:
            print (col)
            obj_list_4.append(col)
            print (pd.unique(temp[col]))

            
            
# obj_list_55 = []
# for col in cols:
#     if (np.array(temp[col]).dtype) == 'object':
#         div = len(pd.unique(temp[col]))
#         if (div<5 ):
#             obj_list_55.append(col)
#             print (col)
#             print (pd.unique(temp[col]))
# print (len(obj_list_55))
num=data.select_dtypes(include=['float64'])
result=pd.concat([temp, num], axis=1)
result.to_csv('newfeaturenight_test.csv',index=False,encoding='utf-8')
