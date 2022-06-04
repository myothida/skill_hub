#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymysql

import pandas as pd
import numpy as np
import random
import base64
import calendar

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from jupyter_dash import JupyterDash
from dash.exceptions import PreventUpdate
from dash import Dash, dash_table
from dash import Input, Output, State, html
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

get_ipython().run_line_magic('load_ext', 'sql')


# In[2]:


get_ipython().run_line_magic('sql', 'mysql://dspuser:dsp@2021!@202.144.157.56/dsp_db')
data = get_ipython().run_line_magic('sql', 'SELECT * FROM student_dtls JOIN students_courses ON students_courses.student_id=student_dtls.student_id JOIN course_master ON course_master.id=students_courses.course_id JOIN dessung_profile ON dessung_profile.cid=student_dtls.cid JOIN dzongkhag_master ON dzongkhag_master.id = dessung_profile.dzongkhag_id JOIN qualifications_master ON qualifications_master.id=dessung_profile.qualification_id;')
t=pd.DataFrame(data)
t.rename(columns={8:'CID #',10:'DOB',11:'Email',14:'Name',27:'Programme',35:'DID',38:'Sex',47:'Dzongkhag',49:'Qualification'},inplace=True)
df=t[['CID #','Name','Programme','DID','Sex','Dzongkhag','Qualification']]


# In[3]:


image1 =  'desuung.png'
encoded_image1 = base64.b64encode(open(image1, 'rb').read())
image2 = 'dsp.jpg'
encoded_image2 = base64.b64encode(open(image2, 'rb').read())
dessup = 'main.png'
dessup = base64.b64encode(open(dessup, 'rb').read())


# # layout

# In[4]:


card_Level = [    
        dbc.CardBody(
            [
                html.H4("Course (by Level)",className="card-title", style = {'font-family': 'Comic Sans MS','color':'#fcba03'}),
                html.Br(),
                html.H5(("Adavanced    : 1"), className="card-title"),
                html.Br(),
                html.H5(("Intermediate : 24"), className="card-title"),
                html.Br(),
                html.H5(("Basic        : 74"), className="card-title"),

            ]
        ),
    ]


# In[5]:


card_status = [    
        dbc.CardBody(
            [
                html.H4("Course (by Status)",className="card-title", style = {'font-family': 'Comic Sans MS','color':'#fcba03'}),
                html.Br(),
                html.H5(("Completed : 77"), className="card-title"),
                html.Br(),
                html.H5(("On-going : 21"), className="card-title"),
                html.Br(),
                html.H5(("Future : 1"), className="card-title"),

            ]
        ),
    ]


# In[6]:


card_student1 = [    
        dbc.CardBody(
            [
                html.H4("Students(Completed)", className="card-title",style={'font-family': 'Comic Sans MS','fontsize':20}),
                dbc.CardImg(src='data:image/jpg;base64,{}'.format(dessup.decode()), top=False,
                            style={'width': '250px','height':'200px','object-fit': 'cover'}),
                html.Div([ 
                         dbc.Button(("Total Student Completed: 1830 "),style={'margin':'1px','textAlign':'left'}),
                        ],style={'display': 'flex','margin-top':'6px','margin-bottom':'4px'}),  
            ],style={'width': '100%'}
        ),
    ]


# In[7]:


card_student2 = [    
        dbc.CardBody(
            [
                html.H4("Students(Completed)", className="card-title",style={'font-family': 'Comic Sans MS','fontsize':20}),
                dbc.CardImg(src='data:image/jpg;base64,{}'.format(dessup.decode()), top=False,
                            style={'width': '250px','height':'200px','object-fit': 'cover'}),
                html.Div([ 
                         dbc.Button(("Total Student Ongoing: 493"),style={'margin':'1px','textAlign':'left'}),
                        ],style={'display': 'flex','margin-top':'6px','margin-bottom':'4px'}),  
            ],style={'width': '100%'}
        ),
    ]


# In[8]:


g = get_ipython().run_line_magic('sql', 'SELECT * FROM course_dtls JOIN course_master ON course_master.id=course_dtls.course_master_id JOIN dsp_centre_master ON dsp_centre_master.id = course_dtls.dsp_centre_id JOIN dzongkhag_master ON dzongkhag_master.id = dsp_centre_master.dzongkhag_id JOIN course_level_master ON course_level_master.id = course_master.course_level_id JOIN department_master ON department_master.id = course_master.department_id;')
op=pd.DataFrame(g)
op.rename(columns={5:'Total_stu',8:'Course Status',10:'End_date',13:'Start Date',24:'Course Name',35:'Dzongkhag',41:'Course Level',47:'Industrial Sector'},inplace=True)
df=op[['Total_stu','Course Status','End_date','Start Date','Course Name','Dzongkhag','Course Level','Industrial Sector']]


# In[9]:


def gen_sum_status(df):
    status_sum_df = df['Course Status'].value_counts().to_frame()
    status_sum_df.reset_index(inplace=True)
    status_sum_df.rename(columns = {'index':'Course Status','Course Status':'Number'},inplace = True)
    
    return status_sum_df


# In[10]:


def gen_sum_demographic(df):
            
    df['Dzongkhag'] = df['Dzongkhag'].str.upper()
    dzo_lat_map = {}
    dzo_log_map = {}
    with open('location_dict.txt') as f:
        for line in f:
            (key, val) = line.split(':')
            dzo_lat_map[key] = val.replace('\n','').strip('[]').replace("'","").split(',')[0]
            dzo_log_map[key] = val.replace('\n','').strip('[]').replace("'","").split(',')[1].strip()    
    
    df['LAT'] = df['Dzongkhag'].map(dzo_lat_map).astype(float)
    df['LON'] = df['Dzongkhag'].map(dzo_log_map).astype(float)
    
    #df['Total_students']=df['Cohort Size (M)']+df['Cohort Size (F)']
    ft = df.groupby(['LAT','LON','Dzongkhag']).count()[['Course Name']]
    ft.reset_index(inplace=True)
    ft.rename(columns={'Course Name':'Number of course'},inplace=True)
    pg = df.groupby(['LAT','LON','Dzongkhag']).sum()[['Total_stu']]
    pg.reset_index(inplace=True)
    demographic_sum_df=ft.merge(right=pg,how='left',on=['LAT','LON','Dzongkhag'])
    
    return demographic_sum_df


# In[11]:


def gen_sum_level(df):
    level_sum_df = df['Course Level'].value_counts().to_frame()
    level_sum_df.reset_index(inplace=True)
    level_sum_df.rename(columns = {'index':'Course Level','Course Level':'Number'},inplace = True)
    
    return level_sum_df


# In[12]:


def gen_status_data(df):
    #df['Total']=df['Cohort Size (M)']+df['Cohort Size (F)']
    stdf = df[['Course Name','Course Status','Industrial Sector','Total_stu']]
    status = stdf.groupby(['Course Status','Industrial Sector']).count()[['Course Name']]
    status.reset_index(inplace=True)

    std = stdf.groupby(['Course Status','Industrial Sector']).sum()[['Total_stu']]
    std.reset_index(inplace=True)

    statusdf = status.merge(right = std,how = 'left',on = ['Course Status','Industrial Sector'])
    statusdf.rename(columns={'Total_stu':'Total Number Of Student','Course Name':'Number of courses'},inplace=True)
    statusdf['Total Number Of Student']=statusdf['Total Number Of Student'].astype(int)
    
    return statusdf


# In[13]:


def gen_level_data(df):
    #df['Total']=df['Cohort Size (M)']+df['Cohort Size (F)']
    levdf = df[['Course Name','Course Level','Industrial Sector','Total_stu']]
    level = levdf.groupby(['Industrial Sector','Course Level']).count()[['Course Name']]
    level.reset_index(inplace=True)

    level1 = levdf.groupby(['Industrial Sector','Course Level']).sum()[['Total_stu']]
    level1.reset_index(inplace=True)

    leveldf = level.merge(right = level1,how = 'left',on = ['Course Level','Industrial Sector'])
    leveldf.rename(columns={'Total_stu':'Total Number Of Student','Course Name':'Number of courses'},inplace=True)
    leveldf['Total Number Of Student']=leveldf['Total Number Of Student'].astype(int)
    
    return leveldf


# In[14]:


def gen_time_data(df):
    
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Months'] = df['Start Date'].dt.month
    df['Years'] = df['Start Date'].dt.year
    df.sort_values(by=['Months'],inplace=True)
    #df['Total_stu'] = df['Cohort Size (M)']+ df['Cohort Size (F)']
    
    coursedata=df[['Months','Course Name', 'Years']]
    coursedata1 = coursedata[coursedata['Years']==2021]
    course1=coursedata1.groupby(['Months','Years']).count()[['Course Name']]
    course1.reset_index(inplace=True)
    course11=course1.astype(int)

    coursedata2 = coursedata[coursedata['Years']==2022]
    course2=coursedata2.groupby(['Months','Years']).count()[['Course Name']]
    course2.reset_index(inplace=True)
    course22=course2.astype(int)

    sectordf = pd.concat([course11, course22])
    sectordf['Months'] = sectordf['Months'].apply(lambda x: calendar.month_abbr[x])

    coursedata=df[['Months','Total_stu', 'Years']]
    coursedata11 = coursedata[coursedata['Years']==2021]
    course11=coursedata11.groupby(['Months','Years']).sum()[['Total_stu']]
    course11.reset_index(inplace=True)
    course12=course11.astype(int)

    coursedata21 = coursedata[coursedata['Years']==2022]
    course21=coursedata21.groupby(['Months','Years']).sum()[['Total_stu']]
    course21.reset_index(inplace=True)
    course23=course21.astype(int)

    sectordf1 = pd.concat([course12, course23])
    sectordf1['Months'] = sectordf1['Months'].apply(lambda x: calendar.month_abbr[x])
    
    timedf = pd.concat([sectordf, sectordf1[['Total_stu']]], axis = 1)
    timedf.rename(columns={'Course Name':'Number_of_course'},inplace=True)
    
    timedf=timedf.astype(str)
    timedf["Month_year"] = timedf[['Months', 'Years']].agg('-'.join, axis=1)
    timedf.set_index('Months',inplace=True)
    timedf.reset_index(inplace=True)
    timedf=timedf.astype(str)
    
    return timedf


# In[15]:


def gen_region_data(df):
    region = df[['Course Name', 'Dzongkhag', 'Total_stu']]
    region1 = region.groupby(['Dzongkhag']).sum()[['Total_stu']]
    region2 = region.groupby(['Dzongkhag',]).count()[['Course Name']]
    region2 = region2.rename(columns = {'Course Name': 'Num_of_courses'})
    regiondf = pd.concat([region2, region1], axis = 1)
    #regiondf.reset_index(inplace=True)
    return regiondf


# In[16]:


def total_completed_std(df):
    sector = df[['Course Status','Total_stu']]
    d = sector.groupby(['Course Status']).sum()[['Total_stu']]
    d.reset_index(inplace=True)
    d.rename(columns = {'index':'Course Status','Total_stu':'Number'},inplace = True)
    
    return d


# In[17]:


def gen_sector_data(df):
    sector = df[['Course Name','Industrial Sector']]
    sector = sector.groupby(['Industrial Sector']).count()[['Course Name']]
    sector.reset_index(inplace=True)
    sector.rename(columns={'Course Name': 'Num_of_courses'},inplace = True)

    sector1=df[['Industrial Sector', 'Total_stu']]
    sectordf1=sector1.groupby(['Industrial Sector']).sum()
    sectordf1.reset_index(inplace=True)
    sectordf = pd.concat([sectordf1, sector[['Num_of_courses']]], axis = 1)
    
    return sectordf


# In[18]:


demographic_df=gen_sum_demographic(df)
status_sum_df = gen_sum_status(df)
level_sum_df = gen_sum_level(df)
regiondf     = gen_region_data(df)
timedf=gen_time_data(df)
statusdf=gen_status_data(df)
leveldf=gen_level_data(df)
sectordf=gen_sector_data(df)


# In[19]:


card_location1 = [    
        dbc.CardBody([ 
            html.H4("Qualifications", className="card-title"),
            html.Br(),
            html.P(("Class XII : (403 Male/298 Female)",),
                className="card-text"),
            html.P(("Bachelors Degree : (191 Male/141 Female)"),
                className="card-text"),
        ],style={'font-family': 'Comic Sans MS','fontsize':20,'color':'#ffff','width': '100%'}),
    ]


# In[20]:


card_location2 = [    
                dbc.CardBody([
            html.H4("Student's Demographics Info", className="card-title"),
          html.Br(),
          html.P(("Trashigang : (79 Male/ 69 Female)"),
                className="card-text"),
          html.P(("Monggar : (54 Male/ 47 Female)"),
                className="card-text"),
        ],style={'font-family': 'Comic Sans MS','fontsize':20,'color':'#ffff','width': '100%'}),
    ]


# In[21]:


def gen_card_nonstops(over_all1,Num_gender, repx2, repx3, repx4 ):

    r2=Num_gender.copy()
    l=r2.iloc[0,2]
    l1=r2.iloc[0,1]

    r2=Num_gender.copy()
    o=r2.iloc[1,2]
    o1=r2.iloc[1,1]
    
    r2=Num_gender.copy()
    op=r2.iloc[2,2]
    op1=r2.iloc[2,1]
    
    repeated2 =[
    dbc.CardBody(
        [
            html.H4("Taken 2 Courses", className="card-title",style={'font-family': 'Comic Sans MS','fontsize':20}),
            html.Div([ 
                     dbc.Button(("Female ",l1),style={'margin':'5px','textAlign':'left'}),
                     dbc.Button(("Male ",l),style={'margin':'5px','textAlign':'left','margin-left':'80px'}),
                    ],style={'display': 'flex'}),
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(dessup.decode()), top=False,style={'width': '100%','height':'290px','object-fit': 'cover'}),
        ]),
    ]
    
    repeated3 =[
    dbc.CardBody(
        [
            html.H4("Taken 3 Courses", className="card-title",style={'font-family': 'Comic Sans MS','fontsize':20}),
            html.Div([ 
                     dbc.Button(("Female ",o1),style={'margin':'5px','textAlign':'left'}),
                     dbc.Button(("Male ",o),style={'margin':'5px','textAlign':'left','margin-left':'80px'}),
                    ],style={'display': 'flex'}),
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(dessup.decode()), top=False,style={'width': '100%','height':'290px','object-fit': 'cover'}),
        ]),
    ]
    
    repeated4 =[
    dbc.CardBody(
        [
            html.H4("Taken 4 Courses", className="card-title",style={'font-family': 'Comic Sans MS','fontsize':20}),
            html.Div([ 
                     dbc.Button(("Female ",op1),style={'margin':'5px','textAlign':'left'}),
                     dbc.Button(("Male ",op),style={'margin':'5px','textAlign':'left','margin-left':'100px'}),
                    ],style={'display': 'flex'}),
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(dessup.decode()), top=False,style={'width': '100%','height':'290px','object-fit': 'cover'}),
        ]),
    ]
    
    return repeated2, repeated3, repeated4


# In[22]:


o = get_ipython().run_line_magic('sql', 'SELECT * FROM student_dtls JOIN students_courses ON students_courses.student_id=student_dtls.student_id JOIN course_master ON course_master.id=students_courses.course_id JOIN dessung_profile ON dessung_profile.cid=student_dtls.cid JOIN dzongkhag_master ON dzongkhag_master.id = dessung_profile.dzongkhag_id JOIN qualifications_master ON qualifications_master.id=dessung_profile.qualification_id;')
t=pd.DataFrame(o)
t.rename(columns={8:'CID #',10:'DOB',11:'Email',14:'Name',27:'Programme',35:'DID',38:'Sex',47:'Dzongkhag',49:'Qualification'},inplace=True)
df=t[['CID #','Name','Programme','DID','Sex','Dzongkhag','Qualification']]


# In[23]:


def split20(s):
    s1 = s.split('2.0')
    return s1
def split30(s):
    s1 = s.split('3.0')
    return s1
def splitdf(s):
    s1 = s.split('-')
    return s1
def splitbr(s):
    s1 = s.split('(')
    return s1
def split1(s):
    s1 = s.split(', 1ST')
    return s1
def splitf1(s):
    s1 = s.split('1')
    return s1
def splitf2(s):
    s1 = s.split('2')
    return s1
def splitf3(s):
    s1 = s.split('3')
    return s1
def splitf4(s):
    s1 = s.split('4')
    return s1
def splitf5(s):
    s1 = s.split('5')
    return s1


# In[24]:


def clean_course_names(df):
    df = df[['Name', 'CID #', 'DID', 'Sex', 'Programme', ]].copy()    
    df['Programme']=df['Programme'].str.upper()
    df['Sex']=df['Sex'].str.upper()
    df['Name']=df['Name'].str.upper()
    cols = df[['Programme']] 
    df1 = df.apply(lambda col: col.str.replace('BAKERY AND PASTRY', 'BAKERY AND PASTRY INTERMEDIATE') 
                   if col.name in cols else col)
    cols = df1[['Programme']] 
    df2 = df1.apply(lambda col: col.str.replace('&', 'AND') if col.name in cols else col)
    df2['Programme']=df2['Programme'].apply(lambda x: split20(x)[0])
    df2['Programme']=df2['Programme'].apply(lambda x: split30(x)[0])
    df2['Programme']=df2['Programme'].apply(lambda x: splitdf(x)[0])
    
    
    df3=df2.replace(to_replace ="E", value ="E-COMMERCE")
    df4 = df3.apply(lambda col: col.str.replace('CULINARY ARTS: MULTICUISINE', 'CULINARY ARTS INTERMEDIATE') 
                    if col.name in cols else col)

    df4['Programme']=df4['Programme'].apply(lambda x: splitbr(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: split1(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: splitf1(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: splitf2(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: splitf3(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: splitf4(x)[0])
    df4['Programme']=df4['Programme'].apply(lambda x: splitf5(x)[0])
    
    df5 = df4.apply(lambda col: col.str.replace('MODERN SCULPTING, POTTERY AND PAINTING', 'MODERN SCULPTURE, POTTERY AND PAINTING') 
                    if col.name in cols else col)
    df5.loc[(df5['Programme']=='CULINARY ARTS '), 'Programme']='CULINARY ARTS MULTICUISINE'
    df5.loc[(df5['Programme']=='ICT ADVANCED: SYSTEM WEBDESIGN '), 'Programme']='SYSTEM WEBDESIGN'
    df5.loc[(df5['Programme']=='ICT ADVANCED: NETWORKING '), 'Programme']='NETWORKING'
    df5.loc[(df5['Programme']=='ICT ADVANCED: DIGITAL MARKETING '), 'Programme']='DIGITAL MARKETING'
    df5.loc[(df5['Programme']=='ICT ADVANCED: MULTIMEDIA '), 'Programme']='MULTIMEDIA'
    df5['Name']=df5['Name'].str.strip()
    df5['Sex']=df5['Sex'].str.strip()
    df5['Programme']=df5['Programme'].str.strip()

    
    return df5


# In[25]:


def gen_nonstop_learners(df):
    over_all = df.groupby(['CID #','Sex']).count()[['Programme']]
    over_all.reset_index(inplace = True)
    over_all.rename(columns = {'Programme':'Num_course'},inplace = True)
    ging = over_all.groupby(['Num_course'])
    over_all = ging.apply(lambda x: x.sort_values(by = ['Sex'], ascending = False))
    over_all.reset_index( drop = True, inplace = True)
    over_all1 = over_all['Num_course'].value_counts().to_frame()
    over_all1.reset_index(inplace = True)
    over_all1.rename(columns = {'index': 'Num_of_courses', 'Num_course': 'Num_of_students'}, inplace = True)
    over_all1
    
    overall_data=over_all[over_all['Num_course']>=2]
    
    over_all = df.groupby(['CID #','Sex']).count()[['Programme']]
    over_all.reset_index(inplace = True)
    over_all.rename(columns = {'Programme':'Num_course'},inplace = True)
    ging = over_all.groupby(['Num_course'])
    over_all = ging.apply(lambda x: x.sort_values(by = ['Sex'], ascending = False))
    over_all.reset_index( drop = True, inplace = True)
    overall_data2=over_all[over_all['Num_course']==2]
    rep_list=list(overall_data2['CID #'])
    repx2 = df[df['CID #'].isin(rep_list)].sort_values('CID #')
   
    over_all = df.groupby(['CID #','Sex']).count()[['Programme']]
    over_all.reset_index(inplace = True)
    over_all.rename(columns = {'Programme':'Num_course'},inplace = True)
    ging = over_all.groupby(['Num_course'])
    over_all = ging.apply(lambda x: x.sort_values(by = ['Sex'], ascending = False))
    over_all.reset_index( drop = True, inplace = True)
    overall_data3=over_all[over_all['Num_course']==3]
    rep_list=list(overall_data3['CID #'])
    repx3 = df[df['CID #'].isin(rep_list)].sort_values('CID #')
    
    over_all = df.groupby(['CID #','Sex']).count()[['Programme']]
    over_all.reset_index(inplace = True)
    over_all.rename(columns = {'Programme':'Num_course'},inplace = True)
    ging = over_all.groupby(['Num_course'])
    over_all = ging.apply(lambda x: x.sort_values(by = ['Sex'], ascending = False))
    over_all.reset_index( drop = True, inplace = True)
    overall_data4=over_all[over_all['Num_course']==4]
    rep_list=list(overall_data4['CID #'])
    repx4 = df[df['CID #'].isin(rep_list)].sort_values('CID #')
   
    genderx2=overall_data2.groupby(['Sex']).count()[['CID #']]
    genderx2.reset_index(inplace=True)
    x2_gender=genderx2.T
    x2_gender.rename(columns={0:'FEMALE',1:'MALE'},inplace=True)
    x2_gender.drop(x2_gender.index[0],inplace=True)
   
    genderx3=overall_data3.groupby(['Sex']).count()[['CID #']]
    genderx3.reset_index(inplace=True)
    x3_gender=genderx3.T
    x3_gender.rename(columns={0:'FEMALE',1:'MALE'},inplace=True)
    x3_gender.drop(x3_gender.index[0],inplace=True)
   
    genderx4=overall_data4.groupby(['Sex']).count()[['CID #']]
    genderx4.reset_index(inplace=True)
    x4_gender=genderx4.T
    x4_gender.rename(columns={0:'FEMALE',1:'MALE'},inplace=True)
    x4_gender.drop(x4_gender.index[0],inplace=True)
   
    Num_gender = pd.concat([x2_gender, x3_gender,x4_gender], axis=0)
    Num_gender.reset_index(inplace=True)
    Num_gender.rename(columns={'index':'Non Stop Learner'},inplace=True)
    Num_gender.loc[0,'Non Stop Learner'] = '2 Times'
    Num_gender.loc[1,'Non Stop Learner'] = '3 Times'
    Num_gender.loc[2,'Non Stop Learner'] = '4 Times'

    return over_all1,Num_gender, repx2, repx3, repx4


# In[26]:


def gen_graph_nonstop(df):
    
    repid=list(df['CID #'])

    t3 = df[df['CID #'].isin(repid)].sort_values('CID #')

    sort=t3.sort_values('Programme')
    regrp = sort.groupby(["CID #",'Sex'])["Programme"].apply(','.join).to_frame()
    regrp.reset_index(inplace=True)
    grouping3=regrp.groupby(['Programme']).count()[['CID #']]
    grouping3.reset_index(inplace=True)
    grouping3.rename(columns={'CID #':'Number of Students'},inplace=True)
    grouping3.sort_values(by = ['Number of Students'], ascending = False,inplace=True)
    top3=grouping3.head(10)
    
    return top3


# In[27]:


over_all1,Num_gender, repx2, repx3, repx4=gen_nonstop_learners(df)
gen_graph_3times=gen_graph_nonstop(repx3)
repeated2, repeated3, repeated4= gen_card_nonstops(over_all1,Num_gender, repx2, repx3, repx4 )
card_2times=repeated2
card_3times=repeated3
card_4times=repeated4


# In[28]:


get_ipython().run_line_magic('sql', 'mysql+mysqldb://dspuser:dsp@2021!@202.144.157.56/dsp_db')
ou = get_ipython().run_line_magic('sql', 'SELECT * FROM student_dtls JOIN students_courses ON students_courses.student_id=student_dtls.student_id JOIN course_master ON course_master.id=students_courses.course_id JOIN dessung_profile ON dessung_profile.cid=student_dtls.cid JOIN dzongkhag_master ON dzongkhag_master.id = dessung_profile.dzongkhag_id JOIN qualifications_master ON qualifications_master.id=dessung_profile.qualification_id;')
tp=pd.DataFrame(ou)
tp.rename(columns={8:'CID #',10:'DOB',11:'Email',14:'Name',27:'Programme',35:'DID',38:'Sex',47:'Dzongkhag',49:'Qualification'},inplace=True)
df=t[['CID #','Name','Programme','DID','Sex','Dzongkhag','Qualification']]

def gen_std_qulitication(df):
    df['Qualification'] = df['Qualification'].astype(str)
    df['Qualification'] = df['Qualification'].str.upper()
    df['Sex'] = df['Sex'].str.upper()
    df['Qualification'] = df['Qualification'].str.strip()
    
    qualification_dict = {}
    with open("qualification_dict.txt") as f:
        for line in f:
            (key, value) = line.split(':')
            qualification_dict[key] = value.replace('\n','').strip('[]').replace("'","").split(',')

    mapping = {}
    for key, values in qualification_dict.items():
        for item in values:
            item = item.strip()
            mapping[item] = key

    df['Qualification'] = df['Qualification'].map(mapping)
    
    return df


# In[29]:


def generate_clean_dz_data(df):
    df['Sex']=df['Sex'].str.upper()
    df['Dzongkhag'] = df['Dzongkhag'].str.upper()
    df.loc[(df['Dzongkhag']=='\tPEMA GATSHEL'), 'Dzongkhag']='PEMA GATSHEL'
    
    dzo_lat_map = {}
    dzo_log_map = {}
    with open('location_dict.txt') as f:
        for line in f:
            (key, val) = line.split(':')
            dzo_lat_map[key] = val.replace('\n','').strip('[]').replace("'","").split(',')[0]
            dzo_log_map[key] = val.replace('\n','').strip('[]').replace("'","").split(',')[1].strip()    
    
    df['LAT'] = df['Dzongkhag'].map(dzo_lat_map).astype(float)
    df['LON'] = df['Dzongkhag'].map(dzo_log_map).astype(float)
    df['LAT'].fillna(27.3,inplace=True)
    df['LON'].fillna(90.3,inplace=True)
    
    std_location_details=df.groupby(['Dzongkhag','LAT','LON','Sex']).count()[['CID #']]
    std_location_details.reset_index(inplace=True)
    std_location_details.rename(columns={'CID #':'Total Number of Student'},inplace=True)
    
    qulidf = df[['CID #','Dzongkhag','Sex']]
    max_student = qulidf.groupby(['Dzongkhag','Sex']).count()[['CID #']]
    max_student.reset_index(inplace=True)
    max_student.rename(columns={'CID #':'Total Number of Student'},inplace=True)
    max_student = max_student.sort_values(by='Total Number of Student',ascending=False)
    
    return std_location_details,max_student


# In[30]:


def gen_qualification_df(df):
    df = gen_std_qulitication(df)
    df['Qualification'] = df['Qualification'].astype(str)
    df['Qualification'] = df['Qualification'].replace(['nan'],'Bachelors Degree')
    qulidf = df[['CID #','Qualification','Sex']]
    qulification = qulidf.groupby(['Qualification','Sex']).count()[['CID #']]
    qulification.reset_index(inplace=True)
    qulification.rename(columns ={'CID #': 'Number_of_students'}, inplace = True)

    return qulification 


# In[31]:


def qualification_card(df):
    df=gen_std_qulitication(df)
    qulidf = df[['CID #','Qualification','Sex']]
    max_qulification = qulidf.groupby(['Qualification','Sex']).count()[['CID #']]
    max_qulification.reset_index(inplace=True)
    max_qulification = max_qulification.sort_values(by='CID #',ascending=False)
    
    return max_qulification


# In[32]:


df.drop_duplicates(subset="CID #",inplace=True)

qulification=gen_qualification_df(df)
max_qulification=qualification_card(df)

std_location_details,max_student=generate_clean_dz_data(df)
std_location_details.loc[std_location_details['Sex']=='M','LAT']=std_location_details.loc[std_location_details['Sex']=='M','LAT']+0.05


# # Main Layout

# In[33]:



app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME])
app.title = "DSP Dashboard"  
server = app.server


# In[34]:


app.layout= dbc.Container([
    html.Div([

        html.Div([
            html.Img(
                    src = 'data:image/png;base64,{}'.format(encoded_image1.decode()),
                    height = '120 px',width = 'auto')
            ],className = 'col-2',
            style = {
                    'align-items': 'center','padding-top' : '4%','height' : 'auto'}), 

        html.Div([
            html.H1(children='De-suung Skilling Programme Dashboard',
                    style = {'font-family': 'Comic Sans MS','textAlign':'center','color':'#fcba03','fontsize':60}
            )],
            className='col-8',
            style = {'padding-top' : '4%'}
        ),

        html.Div([
            html.Img(src = 'data:image/jpg;base64,{}'.format(encoded_image2.decode()),
                    height = '120 px', width = 'auto')
            ],
            className = 'col-2',
            style = {
                    'align-items': 'center','padding-top' : '4%','height' : 'auto'})

        ],
        className = 'row',
        style = {'height' : '4%'}
        ),
    
    
    html.Br(),
    html.Br(),
    
    html.Div([
        dbc.Button('Overall Information',id='course',color='info',size="lg", style={'width':'420px'}),
        dbc.Button('Students Information',id='location',color='info',size="lg",style={'width':'420px'}),
        dbc.Button('Non Stop Learners',id='learn',color='info',size="lg",style={'width':'420px'}),
    ],className="d-grid gap-3 d-md-flex justify-content-md-left", style={'display':'flex','font-family': 'Comic Sans MS'}),                   
             
    
    html.Br(),
    html.Br(),

    html.Div([
            dbc.Collapse([
            html.Div([
                   html.Div([dbc.Col(dbc.Card(card_Level, color="dark", inverse=True,style={'height':'230px'}), width = 3),
                            dbc.Col(dbc.Card(card_status, color="dark", inverse=True,style={'height':'230px'}),style={"margin-left": "10px"}, width = 3),
                            dbc.Col(dbc.Card(card_student1, color="dark", inverse=True,style={'height':'345px'}),style={"margin-left": "10px"}, width = 3),
                            dbc.Col(dbc.Card(card_student2, color="dark", inverse=True,style={'height':'345px'}),style={"margin-left": "10px"}, width = 3),
                           ],style = {'display': 'flex'}),
                   dbc.Row(
                       html.Div([html.H2('Select',style={'font-family': 'Comic Sans MS','textAlign':'left',
                              'color':'#3498db','fontsize':40}),
                       html.Br(),
                       dcc.Dropdown(id='std_id',clearable=False,
                                options=[
                                         {'label':'Timeline','value':'l'},
                                         {'label':'Training Sector','value':'o'},
                                         {'label':'Demographic','value':'lr'},
                                         {'label':'Status','value':'p'},
                                         {'label':'Level','value':'g'}],
                                placeholder='' ,
                                 style={'font-family': 'Comic Sans MS','fontsize':40,
                                        'height':'50px','width': '250px','color':'#9932CC'}),
                             ],style={'margin-left': '50px','margin-top': '-100px','display':'in-block','font-family': 'Comic Sans MS','width':'25%','padding':'4px'}),),

             ],style = {'display': 'block'}),],id='coll',is_open=False),]),
    
    
    
    html.Br(),
    
    html.Div(
        dbc.Collapse(
                html.Div([
                    dcc.Graph('plot1'),
                ],id= 'gaph_id',),id='coll_id',is_open=False,),
    ),

    html.Div([
            dbc.Collapse([
                html.Div([
                    html.Div([
                    html.H2('Select',style={'font-family': 'Comic Sans MS','textAlign':'left','color':'#3498db','fontsize':40}),
                    html.Br(),
                    dcc.Dropdown(id='st',clearable=False,
                                options=[
                                         {'label':'Qualification','value':'j'},
                                         {'label':'Demographic','value':'jp'}],
                                placeholder='' ,
                                style={'font-family': 'Comic Sans MS','fontsize':40,'color':'#9932CC'}),
                   ],style={'display':'in-block','width':'20%',"margin-right": "80px","margin-bottom": "20px"}),
                
                html.Div([
                       dbc.Col(dbc.Card(card_location1, color="dark", inverse=True,style={'height':'170px'}), width = 8,style={"margin-left": "10px"}),
                       dbc.Col(dbc.Card(card_location2, color="dark", inverse=True,style={'height':'170px'}), width = 8,style={"margin-left": "10px"}),
               ],style={'display':'flex'})   
                ],style={'display':'flex'}),  
            ],id='coll1',is_open=False), 
        ]),
    
    html.Br(),   
    
    
    html.Div(
        dbc.Collapse(
                html.Div([
                    dcc.Graph('plot3'),
                ],id= 'gaph_id1'),id='coll_id1',is_open=False),
    ),
    
    html.Div([
            dbc.Collapse([
                    html.Div([
                            dbc.Row([
                                    dbc.Col(dbc.Card(card_2times, color="dark", inverse=True), width = 4),
                                    dbc.Col(dbc.Card(card_3times, color="dark", inverse=True), width = 4),
                                    dbc.Col(dbc.Card(card_4times, color="dark", inverse=True), width = 4),
                                    
                                ],className="mb-4",
                            ),
                          ],style={'display':'flex'}),
                        ],id='coll_id3',is_open=False),
            ]),
    html.Div(
        dbc.Collapse(
              dbc.DropdownMenu(label="Download",menu_variant="dark",id='pwe',size="lg",color='info',
                            children=[
                                dbc.DropdownMenuItem("2 Courses",id="dropdown-button"),
                                dbc.DropdownMenuItem("3 Courses",id="dropdown-button1"),
                                dbc.DropdownMenuItem("4 Courses",id="dropdown-button2")]
                        ,style={"position": "absolute","right": "190px"}),id='ty',is_open=False),),
    
    html.Div(
        dbc.Collapse(
                dbc.Tabs([
                        dbc.Tab(label="Overall", tab_id="r2"),
                        dbc.Tab(label="3 Courses", tab_id="r3"),
                        dbc.Tab(label="4 Courses", tab_id="r4"),
                      ], id="tabs", active_tab="r2",style={'color':'#0a0a08','font-family': 'Comic Sans MS','fontsize':30}
                    ),id='p',is_open=False),style={"display": "flex", "flexWrap": "wrap"}),
    dcc.Download(id="download"),
    dcc.Download(id="download1"),
    dcc.Download(id="download2"),
    html.Br(),
    html.Br(),
    
    html.Div([
        dbc.Collapse(
                html.Div([
                    dcc.Graph('plot4'),
                       ],style={'display':'flex','width':'100%'})
            ,id='coll_id6',is_open=False),
         ]),
                                       
])


# In[35]:


@app.callback(
        Output('coll','is_open'),
        Input('course','n_clicks'),
        State('coll','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[36]:


@app.callback(
        Output('coll_id','is_open'),
        Input('course','n_clicks'),
        State('coll_id','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[37]:


@app.callback(
        Output('coll_id1','is_open'),
        Input('location','n_clicks'),
        State('coll_id1','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[38]:


@app.callback(
    Output('plot1','figure'),
  Input('std_id','value'))

def draw_graph(std):
    if std is None:
        raise PreventUpdate
    
    elif (std=='l'):
        timedf1=timedf[['Number_of_course','Total_stu']].astype(int)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=timedf['Month_year'], y=timedf1['Number_of_course'], name="Courses"),
            secondary_y=False)
        fig.add_trace(
            go.Scatter(x=timedf['Month_year'], y=timedf1['Total_stu'], name="Students"),
            secondary_y=True,)
        fig.update_yaxes(title_text="<b>Number of course</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Number of students</b>", secondary_y=True)
        fig.update_layout(title='Number of Courses and Students on Time basis (2021-2022)',
                               xaxis_title='Months')
        fig.update_layout(autosize = False,
                         width = 1300,
                         height = 500,
                         margin = dict(
                         l=50,
                         r=50,
                         b=100,
                         t=100,
                         pad=4))
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
    
    elif (std=='o'):
        fig = px.bar(sectordf, x="Industrial Sector", y='Total_stu',color="Industrial Sector", title="TOTAL NUMBER OF STUDENTS BY SECTORS")
        fig.update_layout(autosize = False,
                         width = 1300,
                         height = 600,
                         margin = dict(
                         l=50,
                         r=50,
                         b=100,
                         t=100,
                         pad=4))
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    elif (std=='lr'):
        fig= px.scatter_mapbox(demographic_df, lat = 'LAT', lon = 'LON',size_max=30, size='Total_stu',hover_name="Dzongkhag",hover_data={'LON':False,'LAT':False},height=300,color='Number of course',color_continuous_scale='Bluered_r', title='TOTAL NUMBER OF COURSES CONDUCTED IN DIFFERENT REGIONS')
        fig.update_layout(mapbox_style='open-street-map')
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout(autosize = False,
                 width = 1300,
                 height = 800,
                 margin = dict(
                 l=50,
                 r=50,
                 b=100,
                 t=100,
                 pad=4))    
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    elif (std=='g'):
        fig = px.sunburst(leveldf, path=['Course Level','Industrial Sector','Number of courses'],values='Total Number Of Student',
                          color='Course Level',title='NUMBER OF COURSES BY LEVEL AND SECTOR')
        fig.update_layout(autosize = False,width = 1300,height = 800,
                 margin = dict(l=50, r=50, b=100,t=100,pad=4))
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        

    else:
        fig = px.treemap(statusdf, path=['Industrial Sector','Course Status','Number of courses'], values='Total Number Of Student',color='Course Status',title='NUMBER OF COURSES BY SECTOR AND STATUS')
        fig.update_layout(autosize = False,
                 width = 1300,
                 height = 600,
                 margin = dict(
                 l=50,
                 r=50,
                 b=100,
                 t=100,
                 pad=4))
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        
    return fig


# In[39]:


@app.callback(
        Output('coll_id3','is_open'),
        Input('learn','n_clicks'),
        State('coll_id3','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[40]:


@app.callback(
        Output('coll1','is_open'),
        Input('location','n_clicks'),
        State('coll1','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[41]:


@app.callback(
    Output('plot3','figure'),
  Input('st','value'))

def draw_graph(stp):
    if stp is None:
        raise PreventUpdate 
    
    elif (stp=='j'):
        fig2 = px.sunburst(qulification, path=['Sex','Qualification'], values='Number_of_students', title="TOTAL NUMBER OF STUDENTS BY QUALIFICATIONS")
        fig2.update_layout(autosize = False,
                             width = 1300,
                             height = 600,
                             margin = dict(
                             l=50,
                             r=50,
                             b=100,
                             t=100,
                             pad=4))
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    else:
        fig2 = px.scatter_mapbox(std_location_details, lat = 'LAT', lon = 'LON', size='Total Number of Student',color='Sex',color_discrete_sequence=['red','blue'],hover_name="Dzongkhag",hover_data={'LAT':False,'LON':False},title='TOTAL NUMBER OF STUDENT JOIN TO THE COURSE FROM DIFFERENT REGIONS')
        fig2.update_layout(mapbox_style='open-street-map')
        fig2.update_layout(autosize = False,
                             width = 1300,
                             height = 600,
                             margin = dict(
                             l=50,
                             r=50,
                             b=100,
                             t=100,
                             pad=4))
        fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell")) 
        
    return fig2


# In[42]:


@app.callback(
        Output('coll_id6','is_open'),
        Input('learn','n_clicks'),
        State('coll_id6','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[43]:


@app.callback(
        Output('p','is_open'),
        Input('learn','n_clicks'),
        State('p','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[44]:


@app.callback(
    Output('plot4','figure'),
    Input('tabs','active_tab'))

def draw_graph(tabs): 
    if tabs is None:
        raise PreventUpdate
    elif (tabs=='r2'):
        fig4 = px.pie(over_all1, values='Num_of_students', names='Num_of_courses', hole = 0.5)
        fig4.update_layout(title_text="DSP students distribution",
            annotations=[dict(text=str(over_all1['Num_of_students'].sum())+ ' students', 
                              x=0.5, y=0.5, font_size=20, showarrow=False)])
        fig4.update_layout(height=500,width=1300)
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    elif (tabs=='r3'):
        fig4= px.treemap(gen_graph_3times,path=['Programme','Number of Students'],values='Number of Students',color='Programme',title='Students Repeated 3 Courses')
        fig4.update_layout(height=600,width=1300)  
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    else:
        repx4["CID #"]=repx4["CID #"].astype(str)
        fig4 = px.bar(repx4, x="CID #",color='Programme', title="Students Enrolled in Four Courses",text='Programme',
             labels={'count':'Number of Courses','CID #':'CID'})
        fig4.update_layout(showlegend=False,height=400,width=1300)
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        
    return fig4


# In[45]:


@app.callback(
        Output('ty','is_open'),
        Input('learn','n_clicks'),
        State('ty','is_open'),
)

def grawfig(n,is_open):
    if n:
        return not is_open
    return is_open


# In[46]:


@app.callback(
  Output("download", "data"), 
  Input('dropdown-button','n_clicks'),
  prevent_initial_call=True,
)

def draw_graph(re):
    if (re=='dropdown-button'):
        return dcc.send_data_frame(repx2.to_csv, "2courses.csv")
    
    else:
        return dcc.send_data_frame(repx2.to_csv, "2courses.csv")


# In[47]:


@app.callback(
  Output("download1", "data"), 
  Input('dropdown-button1','n_clicks'),
  prevent_initial_call=True,
)

def draw_graph(e):
    if (e=='dropdown-button1'):
        return dcc.send_data_frame(repx3.to_csv, "3courses.csv")
    
    else:
        return dcc.send_data_frame(repx3.to_csv, "3courses.csv")


# In[48]:


@app.callback(
  Output("download2", "data"), 
  Input('dropdown-button2','n_clicks'),
  prevent_initial_call=True,
)

def draw_graph(e):
    if (e=='dropdown-button1'):
        return dcc.send_data_frame(repx4.to_csv, "4courses.csv")
    
    else:
        return dcc.send_data_frame(repx4.to_csv, "4courses.csv")


# In[ ]:


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)    
    url = "http://127.0.0.1:{0}".format(port)    
    app.run_server(use_reloader=False, debug=True, port=port)


# In[ ]:




