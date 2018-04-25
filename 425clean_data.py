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
data = data.rename(columns={"����ѹ": "Systolic", "����ѹ": "Diastolic", "Ѫ���������":"triglyceride", "Ѫ����ܶ�֬����":"HDL", "Ѫ����ܶ�֬����":"LDL"})
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
         '���ֽ�ʱ���'   : np.nan,
         '�������'      : np.nan,
         '������档'     : np.nan,
         '������浥'    : np.nan,
         '���ͼ�ı���'  : np.nan,
         '������浥��'  : np.nan,
         '��ֽ�ʱ��浥'   : np.nan,
         '�����浥'      :np.nan,
         '��ͼ�ı���'    :np.nan,
         '���ֽ�ʱ��浥' :np.nan,
         '���ֽ�ʱ��� ���ֽ�ʱ���' : np.nan,
         '������浥 ������浥' : np.nan,
         '������鵥'    : np.nan,
         'δ��'         : np.nan,
         '����'          :np.nan,
         'δ��'          : np.nan,
        'δ����'            :np.nan,
        'δ���� δ����'      :np.nan,
         '�걾���˼�'     : np.nan,
         '��֬�󸴲�'     : np.nan, # 191
        '������������������档':np.nan,
        '���ҽ�����ơ�'      :np.nan,
        '�����ҽ��������'   :np.nan,


        '����ɣ����鸴���ѯ�绰��52190566ת588':1,

         '-'             :0,
         '--'            :0,
         '- -'           :0,
         '-----'         :np.nan,
         '---'           :np.nan,
         '----'          :np.nan,
         '����'          :0,
         ' ����'         :0,
         '���ԣ�-��'     :0,
         '����(-)'       :0,
         '���� ����'      :0,
         '���� -'         :0,
         '- ����'         :0,
         '���ԣ�-�� ����'  :0,

         '+'             :2,
         '����'          :2,
         '����(+)'       :2,
         '���ԣ�+��'     :2,
         '�R1:80����'     :2,
          '���'         :2,
         '������'        :1,
         '����'          :1,  #  300036
         '������(��)'     :1,  #  300036
         '++'           :3,
         '+++'          :4,
         '++++'         :5,
         '��'           :2,
         '����'          :3,
         '������'        :4,
         '����(���)'    :2,
         '����(�ض�)'    :4,
         '����(�ж�)'    :3,

         '+-'           :2,
         '����'         :2,
         '���� +'       :2,
         '����'         :3,

         '�ټ�'          :1,
         '���'          :2,

         '����'          :0,
         '���� ����'      :0,
         'δ���쳣'      :0,
         'δ�������쳣'   :0,
         'δ�����쳣'     :0,
         '��������'       :0,
         'δ�����쳣, δ�����쳣' :0,
         'δ��'           :np.nan,
         'δ���쳣 δ���쳣' :0,
         'δ�� δ��'      :0,
         'ȫ�Ĺ���δ���������쳣' :0,
         '���Ǻ���δ���쳣' :0,
         'δ����'        :0,
         'δ���'        :0,
         'δ���쳣 δ���쳣 δ���쳣':0,
         '���� ���� ���� ����' :0,
         '���� ���� ���� ���� ����':0,
         '���� ���� ����' :0,
         '�쳣'          :2,

         '�� ��'         :0,
         'δ��'          :0,
         '��'            :0,

         '���(S)'       :1,
         '��ҩ(R)'       :2,
         '��ҩ'          :2,
         '�ж����(MS)'  :3,
         '���'          :4,

         '��'            :1,
         '��'            :2,
         '��'            :3,
         '��'            :4,

         '1+'            :1,
         '+1'            :1,
         '3+'            :3,
         '2+'            :2,

         '����о�'       :0,
         '�н�'           :1,
         '����'           :2,

         '5��'            :5,
         '4��'            :4,
         '3��'            :3,
         '2��'            :2,
         '1��'            :1,
         '0��'            :0,

         'δ��⵽ȱʧ��'   :0,
         'δ��⵽ȱʧ'     :0,
         '����ȱʧ'        :1,
         'δ��⵽ͻ�䡣'   :0,
         '����ͻ�䣬IVS-II-654λ��ͻ������Ӻ��ӡ�' :2,

         '�����½�'        :1,
         '�½�'            :1,
         '�Ҷ������½�����' :1,

         'HIV��������'     :0,
         'HIV��������(-)'   :0,
         'HIV��Ⱦ��ȷ��'    :1,
         '������'           :1,

         'yellow'          :1,
         '��ɫ'            :1,
          '����ɫ'         :2,
          'ǳ��ɫ'         :2,
         '��ɫ'            :3,
         '�ƺ�ɫ'          :4,
         '���ɫ'          :4,
         '����ɫ'          :5,
          '����'           :6,
         '��ɫ'            :7,
         '����ɫ'          :8,
         '����ɫ'          :9,
          '��ɫ'           :8,
         '��ɫ'            :10,
         '����'            :11,

         '͸��'            :0,
         '����'           :1,
         '����'            :1,

         '��'              :1,
         '��,��״'          :2,
         '��ϡ��'          :3,
         'ϡ'              :4,
          '��'             :5,
          'Ӳ'             :6,

         'O'               :1,
         'O��'             :1,
         '��O����'           :1,
         '��O��'             :1,
         'O ��'            :1,
         'O  ��'           :1,
         '0'               :1,
         '0��'             :1,
         'O��Ѫ'           :1,
         'A��'             :2,
          'a��'            :2,
         'A'               :2,
         '(A)'             :2,
          '��A����'          :2,
          '��A��'            :2,
          'A  ��'          :2,
          'B��'            :3,
          'B'              :3,
        '��B����'            :3,
         'B  ��'           :3,
           'B ��'          :3,
         'b��'             :3,
         'AB'              :4,
         'AB��'            :4,
         'AB ��'           :4,
         '��AB����'          :4,
          '(AB)'           :4,
          '��AB��'           :4,

         '���񾭶�λ����'    :1,
         '������֫��������'  :2,
         '��֫��������'      :3,
         '���񾭶�λ���� ���񾭶�λ����' :4,

         '��ѹʹ��'         :0,
         'ߵ��ʹ'           :1,
         'ѹʹ'             :2,
         'ߵ��ʹ, ߵ��ʹ'    :1,
         'ѹʹ, ߵ��ʹ'      :3,

         'Ů������ָ��'     : -1, # 300076

         '4.03 4.03'       :4.03,
         '2.1.'            :2.1,
         '9.871 9.87'      :9.871,
         '36.0 36.0'       :36.0,
        '2..99'            :2.99,
         '44.7 44.7'       :44.7,
         '����1.496'       :1.496,
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
         '3��89'           :3.89,
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
         '0.01�����ԣ�'     :0.01,
         '/'               :np.nan,
         '3.4~33.9'        :3.4,
         '2.6 2.9'         :2.6,
         '��250.00'        :250,
         '5.32 4.39'       :5.32,
         '9.70 11.41'      :9.7,
         '78.02 89.02'     :78.02,
          '1.84 1.93'      :1.84,
         '10.3 14.6'       :10.3,
         '6.93 9.67'       :6.93,
         '0':0,
        '0':0,
        '0-1':1,
        '�Ӳ���':0,
        '����': 1,
        '/'  : np.nan,
        'δ�� 1':np.nan,
        '10^4':6,
        'S':5,
        '4':4,
        '2-5':4,
        '��Ƥϸ������':1,
        'ճҺ˿+' :2   ,
        '���񾭶�λ���� ���񾭶�λ����':0,
        '���񾭶�λ����':0,
        '��������ڣ�������δ����':1,
        '˫�������ڳ�Ѫ':1,
        '������������״�':1,
        'δ����':1,
        '���ֽ�ʱ��浥���ѯ���Ӱ棨http://www.xyzjkjt.com/��':np.nan,
        '�������䱨���ѯ���¼���Ӱ���ַ��http://www.xyzjkjt.com/���˺ţ���������           ���룺����������  ����19880810��':np.nan,
        '������Ƭ��':np.nan,
        '��������ҽ�����ⱨ��':np.nan,
        'δ��ʾ':np.nan,
        'δ����':0,
        '��ҽ���ʱ�ʶδ���������쳣':0,
        '����':1,
        '1.0 1.0':1,
        '���Ժ���':1,
        '���':1,
        '��100':80,
        '��ȵ����򲡻�����ǰ�ڷ��գ��ǻ���������ƫ�ߣ�':1,
        '���򲡻�����ǰ�ڷ��ս�С':1,
        '��':1,
        '1:100' :1,
        '1:320 +':3,
        '1:100 +':1,
        '������':np.nan,
        '����I��,δ���쳣':0,
        '�ͼ첣Ƭһ�ţ����죺������Χ�ڣ�δ����ϸ����':1,
        '�ͼ첣Ƭһ�ţ����죺��Ƥϸ�������٣����鸴�顣':1,
        '�ͼ첣Ƭһ�ţ����죺��Ƥϸ�������٣����鸴�顣':1,
        '���̣����Ⱦ�':1,
        '����Ժ�ϸ��':1,
        'δ��ʾ':0,
        '�Ǿ�һ��':2,
        'δ��':0,
        'Ƣ���г�':1,
        '�Ӳ���':0,
        '�У������һ��ר�Ƽ��':1,
        '����':0,
        '����':1,
        '���Ժ���':1,
        '�ɼ�ճҺ˿':1,
        '����Һ���ͼ�ǳ��ɫҺ��Լ40ml��TCT��Ƭ�����죺��������״��Ƥ����·��Ƥϸ����δ����ϸ����':0,
        '����Һ���ͼ��ɫҺ��Լ40ml�����ĳ���TCT��Ƭ������Ⱦɫ�����£�����������·��Ƥϸ����δ�ҵ���ϸ����':0,
        '����Һ���ͼ��ɫҺ��Լ40ml�����ĳ���TCT��Ƭһ�ţ�����Ⱦɫ�����£�����������·��Ƥϸ����δ�ҵ���ϸ����':0,
        '����Һ���ͼ��ɫҺ��Լ45ml��TCT��Ƭ�����죺������״��Ƥϸ����������·��Ƥϸ����δ����ϸ����':0,
        'exit':0,
        'I':1,
        'ή��������':1,
        '������':2,
        '+/HP':np.nan,
        '��20':16,
        '0':0,
        '3-5':4,
        '���':np.nan,
        'ʮ��ָ�������޲�������':1,
         '���ۣ���Ĥ0.6mm��ǰ��2.7mm����״��3.8mm��������14.9mm������24mm�����ۣ���Ĥ0.7mm��ǰ��2.6mm����״��3.5mm��������14.6mm������23mm��':1,
        '0-3':2,
        'R':3,
        '#NAME?':np.nan,
        '<1.00E+03':1,
        '0-2':1,
    '#NUM!':np.nan,
    '2+/HP':2,
'11��1��':np.nan,
    '���':np.nan,
    'ָ��':np.nan,
'΢������':np.nan,
'#NUM!'    :np.nan,
'0-5':    4,
 '0-4':3,   
'3.05 3.05':3.05,
    
    '0.96 0.78':0.8,
     '1..27':1.27,
     '0.04 0.04' :0.04,  
    '��':0,
    '50.4 50.4':50.4,
 '79.6 79.6':79.6,
    '���ڸ��ṹδ�������쳣':np.nan,
'֬����':80,
 '6.31.0.45'   :6.31,
'δ�� 1.0':np.nan,
'***.**':np.nan,
'���� ����':np.nan,
'����(��ˮƽ)':np.nan,
'- +':np.nan,
'��0.30':0.0,
'����'     :np.nan, 
    '**.*' :np.nan, 
'60 ����':60,
'fo':np.nan,
         }

mapping2 = {
    # 1321
    '0.4 nan nan nan' : 0.4,
    'δ�� δ��': np.nan,
    'nan nan nan nan': np.nan,
    'nan nan nan': np.nan,
     'nan 1.2 nan nan nan':1.2,
    'nan 1.0':1,
    '1.0 nan':1,
    'nan nan 1.5':1.5,
    'ʧ��' : 0,
    '�ֶ�':np.nan,
         '����':np.nan,
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
    '�޹��':0,
    'nan 0.3 nan':0.3,
    '0.8 nan nan':0.8,
    'nan 1.0 nan':1.0,
    'nan 0.05 nan':0.05,
'nan nan 0.3':0.3,
'δ�� 1.2':1.2,
'nan nan nan nan 1.0':1.0,
'0.7 0.7 0.7':0.7,
'nan nan nan nan nan':np.nan,
'1.0 nan nan':1.0,
'0.7 0.7 0.7 0.7':0.7,
'0.2 0.2 0.2':0.2,
'1.0 nan nan 1.0 1.0':1.0,
'nan 0.5 nan':0.5,
'δҪ���� ':np.nan,
'nan 0.6 nan':0.6,
'nan nan 0.7':0.7,
'0.6 nan nan nan nan':0.6,
'0.6 0.6 0.6 0.6 0.6':0.6,
'0.8 δ��':0.8,
'nan 1.5 nan nan':1.5,
'δ�� 0.5':0.5,
'0.9 0.9 0.9 0.9':0.9,
'δ�� 0.3':0.3,
'0.4 nan nan':0.4,
'0.3 nan nan':0.3,
'nan 0.15 nan':1.5,
'nan 0.9 nan':0.9,
'nan 1.5 nan':1.5,
'nan ����':np.nan,
'nan nan 0.5':0.5,
'���޷���ϲ��ܼ��':np.nan,
'nan 0.8 nan':0.8,
'0.7 nan nan':0.7,
'nan nan 0.6':0.6,
'0.9 nan nan':0.9,
'δҪ����':np.nan,
'0.3 0.3 0.3':0.3,
'0.5 0.5 0.5':0.5,
'0.6 0.6 0.6 0.6':0.6,
'1.2 1.2 1.2 1.2':1.2,

'���ԣ�-�� ���ԣ�-��':1,
    #'424'
    '��������':80,
    '����Ķ�����':56,
    '����Ķ�����':106,
    '��/��' : np.nan,
             # 2229
         '������'            :1,
          '����(��ˮƽ)'      :1,
          '�ض�'             :10,
    '�ֲ�':np.nan,
'����':np.nan,
'����':np.nan,
'���쳣':np.nan,
    '����':60,
        'CM':np.nan,
        '97cm'
        'cm':np.nan,
        '88CM':88,
        '91cm':91,
        '82cm':82,
        '72CM':72,
        '77cm':77,
        '82cm':82,
        '-        0umol/L':0,
        'Normal':np.nan,
        '0(-)':0,
        '8.6(+1)':8.6,
        'II':2,
        'III':3,
        '���':2,
        '΢��':1,
        '���':3,
        '���':4,
        '��TCT':np.nan,
        '��TCT':np.nan,
        '�����TCT':np.nan,
        '����Ƭ':np.nan,
        'iii��':3,
        'ii��':3,
        '���':3,
        '��v':3,
        '���':3,
        '���':3,
        '�ж�':2,
        '���':4,
        '�� ��':2,
        '���':3,
        '��δ���쳣':np.nan,
        'nan':np.nan,
    '����+':2,
    '+++ -':2,
    '- +-':np.nan,
    'ż��':np.nan,
    '����':np.nan,
        '��ĸ��ϸ��++':np.nan,
        '��TCT':np.nan,
        '1��3��':np.nan,
        '�����TCT':np.nan,
        '����Ƭ':np.nan,
        '���':np.nan,
        '�쵽':np.nan,
        '2��4��':np.nan,
       '0':0,
       '4':4,
}


with timer ("mapping ..."):
    temp = temp.applymap(lambda x : mapping[x] if x in mapping.keys() else x )
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('>') else x)
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('<') else x)
    # temp = temp.applymap(lambda x : x[1:] if str(x).startswith('��') else x)
    temp = temp.applymap(lambda x : x[:-1] if str(x).endswith('.') else x)
    temp = temp.applymap(lambda x : mapping2[x] if x in mapping2.keys() else x )

obj_list_4 = []
obj_list = []

unit_mapping = ['kpa', 'db/m', '(ng/mL)', '(pmol/L)', '(U/ml)', '%', '��','kg','(umol/L)']
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
        if  '��' in x :
            return float(x[x.index('��')+1:])
        if  '��=' in x :
            return float(x[x.index('��=')+1:])
        if  '\t' in  x :
            return float(x[x.index('\t')+1:])
        if len(x.split(sep='.'))>2:#2.2.8
            i=x.rindex('.')
            x=x[0:i]+x[i+1:]
            return float(x)
        if 'δ��' in x or 'δ��' in x or '����' in x:
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
    p = re.compile(r'���� (\d.*)')
    q = re.compile(r'���� (\d.*)')
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
    p = re.compile(r'.*(\d{2,3})��/��.*')
    m =  p.match(x)
    q = re.compile(r'.*�Ķ�����(\d{2,3})')
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

dealcol=[1325,  425 , 437 ,3191 , 547 , 1321,  3203,  2233,  3485 , 30007 , 549, 424  ,459101 , 2229 ,901  ,1322 ,1326 ,3429 ,3430 , 459102 , 3194  ,3198 , 733, 212 , 2302]
dealcol=[str(i) for i in dealcol]
unhealth=temp[dealcol].select_dtypes(include=['object'])
for i in unhealth.columns:
    print(i+'*****')
    print(unhealth[i].unique())
dropcol=[547,   2302,  733,]
temp=temp.drop(dropcol,axis=1)

num=data.select_dtypes(include=['float64'])
result=pd.concat([temp, num], axis=1)
result.to_csv('newfeaturenight_test.csv',index=False,encoding='utf-8')           
   