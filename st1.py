import streamlit as st
import pandas as pd
import numpy as np
import xlrd
import xlsxwriter
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn import datasets, neighbors
import base64
from io import BytesIO
import cx_Oracle
import re
#import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
#from sqlalchemy import types, create_engine
import urllib.request

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import RobustScaler

import pickle

#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection






# Headings

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style='text-align: center; color: Light gray;'>Data Processing App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Light gray;'>from Omega to Alpha </h3>", unsafe_allow_html=True)

st.markdown(
"""
<style>
.reportview-container {
    background: url("https://media.giphy.com/media/nIPpfk1KbH27KUvyxH/giphy.gif")
}
.sidebar .sidebar-content {
    background: url("https://media.giphy.com/media/nIPpfk1KbH27KUvyxH/giphy.gif")
}
</style>
""",
unsafe_allow_html=True
)



# Enter the path here where all the temporary files will be stored
temp='\\temp.csv'
temp1='\\temp1.csv'
#os.chdir(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit')
path=os.getcwd()
path1=os.getcwd()
path=path+temp
path1=path1+temp1
#path=(r"C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv")





    
    
    
    
    
    
    
    # All Functions
def mvt_mean(df):
    
    try:
    
        clean_df=(df.fillna(df.mean()))
        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(clean_df.dtypes)
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        st.info("Data to be treated using MEAN : {}".format(list(dict(df.mean()).keys())))
        st.info('Shape of dataframe (Rows, Columns):{} '.format(df.shape))
        st.write('Data description : ',df.describe())
        st.info("Only Numerical missing values will be treated using MEAN ")
        st.info("categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.line_chart(clean_df)
        return clean_df
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    

        
        
        
        
    
def get_table_download_link_csv(df):
    try:
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframes
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(
            csv.encode()
        ).decode()  # some strings <-> bytes conversions necessary here
        return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



def to_excel(df):
    try:
        
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



def get_table_download_link_xlsx(df):
    try:
        
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        val = to_excel(df)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="dataprep.xlsx">Download xlsx file</a>' # decode b'abc' => abc
    

    
    
    
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


    
    
    
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href







    

def mvt_median(df):
    try:
    
        clean_df=(df.fillna(df.median()))
        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(df.dtypes)
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        st.info("Data to be treated using MEDIAN : {}".format(list(dict(df.median()).keys())))
        st.info('Shape of dataframe (Rows, Columns):{} '.format(df.shape))
        st.write('Data description : ',df.describe())
        st.info("Only Numerical missing values will be treated using Median ")
        st.info("categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.line_chart(clean_df)
        return clean_df
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
    
    

    
    

def mvt_mode(df):
    try:
        cat_col=list(df.select_dtypes(include ='object').columns)
        st.info("The Percentage of Value Missing in Given Data is : {:.3f}%".format((df[cat_col].isna().sum().sum())/(df.count().sum())*100))
        st.info("\nThe Percenatge of Value Missing in Given Data is :\n{}".format((df[cat_col].isnull().sum()*100)/df.shape[0]))
        clean_df=(df.fillna(df.select_dtypes(include ='object').mode().iloc[0]))
        st.dataframe(clean_df)
        st.info("\nData to be treated using MODE : {}".format(cat_col))
        st.write('Shape of dataframe (Rows, Columns): ',df.shape)
        st.write('Data description :\n',df.describe(include ='object'))
        st.info("Only categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.info("You can head to Mean or Median to treat the Numerical Missing Value")
        st.line_chart(clean_df)
        return clean_df
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        st.write("This can happen if there is no categorical data to treat")
        return df




def ot_iqr(df,column_name):
    
    try:
        
        
    
        #column_name="Marks_Grad"

        if column_name:



            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            IQR = q3 - q1
            lower_limit = q1 - 1.5*IQR
            upper_limit = q3 + 1.5*IQR
            removed_outlier = df[(df[column_name] > lower_limit) & (df[column_name] < upper_limit)]   
            st.dataframe(removed_outlier)
            st.write("Percentile Of Dataset :\n ", df.describe())
            st.info('Size of dataset after outlier removal')
            st.write(removed_outlier.shape)
            st.line_chart(removed_outlier)
            return removed_outlier
        
        
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

    
    
    
def z_score(df,column_name):
    
    try:
        
    
        if column_name:

            df['z-score'] = (df[column_name]-df[column_name].mean())/df[column_name].std() #calculating Z-score
            outliers = df[(df['z-score']<-1) | (df['z-score']>1)]   #outliers
            removed_outliers = pd.concat([df, outliers]).drop_duplicates(keep=False)   #dataframe after removal 
            st.dataframe(removed_outliers)
            st.write("Percentile Of Dataset :\n ", df.describe())
            st.write('Number of outliers : {}'.format(outliers.shape[0])) #number of outliers in Given Dataset
            st.info('Size of dataset after outlier removal')
            st.write(removed_outliers.shape)
            st.line_chart(removed_outliers)
            return removed_outliers
        
        
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


    
    
        

    
    
    
    

    
    
    
    
    
    
    
    
    



def mvt_knn(df):
    try:
        
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        num_col =list(df.select_dtypes(include='float64').columns)
        knn =KNNImputer(n_neighbors =1,add_indicator =True)
        knn.fit(df[num_col])
        knn_impute =pd.DataFrame(knn.transform(df[num_col]))
        df[num_col]=knn_impute.iloc[:,:df[num_col].shape[1]]
        clean_df= df
        clean_df=(df.fillna(df.mode().iloc[0]))
        st.dataframe(clean_df)
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.info("Numerical data : {}".format(list(dict(df.median()).keys())))
        st.info("Categorical data : {}".format(list(df.select_dtypes(include='object').mode())))
        st.write('Shape of dataframe (Rows, Columns): ',df.shape)
        st.write('Data description : ',df.describe())
        st.line_chart(clean_df)
        st.info("Only Numerical Data is treated using K-NN Method , Categorical Data is trreated using Mode")
        return clean_df
    
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df





def f_ss(df):
    try:
        
        X = df.select_dtypes(include=np.number)
        mean_X = np.mean(X)
        std_X = np.std(X)
        Xstd = (X - np.mean(X))/np.std(X)
        st.dataframe(Xstd)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xstd.shape)
        st.write('Data Informations :',Xstd.info())
        st.write('Data description : ',Xstd.describe())
        st.line_chart(Xstd)
        return Xstd
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


def f_mm(df):
    try:
        
        X = df.select_dtypes(include=np.number)
        min_X = np.min(X)
        max_X = np.max(X)
        Xminmax = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
        st.dataframe(Xminmax)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xminmax.shape)
        st.write('Data Informations :',Xminmax.info())
        st.write('Data description : ',Xminmax.describe())
        st.line_chart(Xminmax)
        return Xminmax
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df





def f_rs(df):
    try:
        
        X = df.select_dtypes(include=np.number)
        median_X = np.median(X)
        q3=X.quantile(0.75)-X.quantile(0.25)
        Xrs =(X - np.median(X))/q3
        st.dataframe(Xrs)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xrs.shape)
        st.write('Data Informations :',Xrs.info())
        st.write('Data description : ',Xrs.describe())
        st.line_chart(Xrs)
        return Xrs
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

    



def maxabs(df):
    try:
        
        X = df.select_dtypes(include=np.number) 
        max_abs_X = np.max(abs(X)) 
        Xmaxabs = X /np.max(abs(X))
        st.dataframe(Xmaxabs)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xmaxabs.shape)
        st.write('Data Informations :',Xmaxabs.info())
        st.write('Data description : ',Xmaxabs.describe())
        st.line_chart(Xmaxabs)
        return Xmaxabs

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


    




    
    
        
 
    
    

    
                
                
            
                    
# MVT Options 


def mvt_options(df):
    
    try:
        
    
        optionm=("Mean","Median","Mode","K NN Imputer")
        mvt_selection=st.sidebar.radio('Choose a Missing Value Treatment Method',optionm)
        if mvt_selection == 'Mean':
            st.sidebar.write('you selected mean')
            if st.sidebar.button('Process Mean'):
                df = pd.read_csv(path)
                df=mvt_mean(df)
                df.to_csv(path, index=False)
                df.to_csv(path1, index=False)
                return df
                #st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        elif mvt_selection == 'Mode':
            st.sidebar.write('You selected mode')
            if st.sidebar.button('Process Mode'):
                df = pd.read_csv(path)
                df=mvt_mode(df)
                df.to_csv(path, index=False)
                df.to_csv(path1, index=False)
                return df







        elif mvt_selection == 'Median':
            st.sidebar.write('You selected Median')
            if st.sidebar.button('Process Median'):
                df = pd.read_csv(path)
                df=mvt_median(df)
                df.to_csv(path, index=False)
                df.to_csv(path1, index=False)
                return df





        elif mvt_selection == 'K NN Imputer':
            st.sidebar.write('You selected K NN Imputer')
            if st.sidebar.button('Process K NN'):
                df = pd.read_csv(path)
                df=mvt_knn(df)
                df.to_csv(path, index=False)
                df.to_csv(path1, index=False)
                return df
            
            

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

            
            

# Outliers Function

def outlier_function():
    try:
        
        option_o=("IQR","Z-Score")
        o_f_selection = st.sidebar.radio("Choose a Outlier Treatment Method",option_o)
        if o_f_selection == "IQR":
            df = pd.read_csv(path)
            column_name=st.selectbox("Please Choose a column from which the outlier will be removed",df.columns)
            #st.info("You can find the list of columns below")
            #st.write(df.columns)
            if st.sidebar.button("Process IQR"):
                df = pd.read_csv(path)
                if column_name in df.columns:

                    df=ot_iqr(df,column_name)
                    df.to_csv(path, index=False)
                    df.to_csv(path1, index=False)
                    return df
                else:
                    st.info("This Column Name is Not Present")

        elif o_f_selection == "Z-Score":
            
           # st.info("You can find the list of columns below")
            df = pd.read_csv(path)
            column_name=st.selectbox("Please Choose a column from which the outlier will be removed",df.columns)
            #st.write(df.columns)
            if st.sidebar.button("Process Z-Score"):
                df = pd.read_csv(path)
                if column_name in df.columns:

                    df=z_score(df,column_name)
                    df.to_csv(path, index=False)
                    df.to_csv(path1, index=False)
                    return df
                else:
                    st.info("This Column Name is Not Present")

                    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

        
        
    
    

    
# feature Scaling options



def fso(df):
    
    try:
        
    
        fs_option=("Standard Scalar","Min Max Scalar", "Max Absolute Scalar" , "Robust Scalar")
        fs_selection=st.sidebar.radio('Choose a Feature Scaling Method',fs_option)


        if fs_selection == 'Standard Scalar':
            st.sidebar.write('you selected Standard Scalar')
            if st.sidebar.button('Process SS'):
                df = pd.read_csv(path)
                df=f_ss(df)
                df.to_csv(path, index=False)
                return df



        elif fs_selection == 'Min Max Scalar':
            st.sidebar.write('you selected min max')
            if st.sidebar.button('Process mm'):
                df = pd.read_csv(path)
                df=f_mm(df)
                df.to_csv(path, index=False)
                return df


        elif fs_selection == 'Max Absolute Scalar':
            st.sidebar.write('You selected max absolute')
            if st.sidebar.button('Process Ma'):
                df = pd.read_csv(path)
                df=maxabs(df)
                df.to_csv(path, index=False)
                return df


        elif fs_selection == 'Robust Scalar':
            st.sidebar.write('You selected Robust Scalar')
            if st.sidebar.button('Process rs'):
                df = pd.read_csv(path)
                df=f_rs(df)
                df.to_csv(path, index=False)
                return df
            

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

            
    
    
    
    
    
def upload_xlsx(uploaded_file):
    
    try:
        
    
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            df.to_csv(path1,index=False)
            return df
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

def upload_csv(uploaded_file):
    
    try:
        
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            df.to_csv(path1,index=False)
            return df

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    
    
    
def mail():
    
    try:
        
        
        mail_content = '''Hello,
        This is a Data Pre Processed File.
        Please see the attachmet below .
        Thank You for using our app
        '''

        #os.chdir(path)
        #The mail addresses and password
        file_name='pass.txt'
        if os.path.exists(file_name):
            with open('pass.txt', 'r') as file:  
                sender_pass=file.read()
                file.close()

        else:
            urllib.request.urlretrieve("https://drive.google.com/u/0/uc?id=1tan_wJsUqOtBTJv1lrwpqqJYgdVJY1td&export=download", "pass.txt")
            with open('pass.txt', 'r') as file: 
                sender_pass=file.read()
                file.close()

        sender_address = 'dpreprocessing@gmail.com'
        regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
        receiver_address = st.text_input("Please Enter The Email Address")
        if receiver_address:
            if(re.search(regex,receiver_address)):
                #Setup the MIME
                message = MIMEMultipart()
                message['From'] = sender_address
                message['To'] = receiver_address
                message['Subject'] = 'Please see your processed file in attachment'
                #The subject line
                #The body and the attachments for the mail
                message.attach(MIMEText(mail_content, 'plain'))
                attach_file_name = path
                attach_file = open(attach_file_name) # Open the file as binary mode
                payload = MIMEBase('application', 'octate-stream')
                payload.set_payload((attach_file).read())
                encoders.encode_base64(payload) #encode the attachment
                #add payload header with filename
                payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
                message.attach(payload)
                #Create SMTP session for sending the mail
                session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
                session.starttls() #enable security
                session.login(sender_address, sender_pass) #login with mail_id and password
                text = message.as_string()
                session.sendmail(sender_address, receiver_address, text)
                session.quit()
                st.write('Mail Sent Successfully to {}'.format(receiver_address))

            else:
                st.warning("Please Enter a Valid Email Address")



    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    
            
            
            
            
            
# File Upload
def file_upload():
    
    try:
        st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data import</h3>", unsafe_allow_html=True)
        
    
        f_option=('.Xlsx','.Csv','Oracle')
        f_select=st.sidebar.radio('Choose a file type',f_option)

        if f_select == '.Xlsx':

            uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")

            if uploaded_file:

                if st.sidebar.button('Upload File'):
                    df=upload_xlsx(uploaded_file)
                    return df



        elif f_select == '.Csv':
            uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

            if uploaded_file:
                if st.sidebar.button('Upload File'):
                    df=upload_csv(uploaded_file)
                    return df

        elif f_select == 'Oracle':

            st.info("Enter Oracle Database information")

            user=st.text_input("Enter User name ")
            passwd=st.text_input("Enter Password ", type="password")
            host=st.text_input("Enter Host Address")
            port=st.text_input("Enter Port number")
            query =st.text_input("Enter the query for the desired data")


            if st.button("Connect"):
                
               # muzzammil/123@46:99/ORCL


                con_query="{}/{}@{}:{}/ORCL".format(user,passwd,host,port)

                con=cx_Oracle.connect(con_query)

                if con!=None:
                    st.info("Connection Established Successfully")
                    df = pd.read_sql(query,con)
                    st.dataframe(df)
                    df.to_csv(path, index=False)
                    return df


                    #query =st.text_input("Fire the query for the desired data")
                    #if st.button("Fire"):
                     #   df = pd.read_sql(query,state.con)
                      #  st.dataframe(df)
                       # df.to_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv', index=False)
                        #return df


        
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
        
    
        

# Data export

def data_export(df):
    
    try:
        
    
    
        
        st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data Export</h3>", unsafe_allow_html=True)
        fd_option=('.Xlsx','.Csv','Oracle','Email')
        fd_select=st.sidebar.radio('Choose a file type to download',fd_option)

        if fd_select == '.Csv':
            if st.sidebar.button('Download Csv'):
                df = pd.read_csv(path)

                st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
                return 0


        elif fd_select == '.Xlsx':
            if st.sidebar.button('Download Xlsx'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
                return 0


        elif fd_select == 'Oracle':
            st.info("Enter Oracle Database information")

            users=st.text_input("Enter Users name ")
            passwd=st.text_input("Enter Password ", type="password")
            host=st.text_input("Enter Host Address")
            port=st.text_input("Enter Port number")
            table=st.text_input("Enter the name of table to create, if table exist it'll be replaced")
            if st.button("Connect"):
                df = pd.read_csv(path)
                conn = create_engine('oracle+cx_oracle://{}:{}@{}:{}/ORCL'.format(users,passwd,host,port))
                df.to_sql('{}'.format(table), conn, if_exists='replace')
                #con_query="{}/{}@{}:{}/ORCL".format(user,passwd,host,port)
                #con=cx_Oracle.connect(con_query)
                if conn!=None:
                    st.info("Connection Established Successfully and Table Inserted")



        elif fd_select == "Email":
            mail()


    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    
    
    
    
    
# EDA Function


def eda(df):
    
    eda_options = ['Pair plot','Histogram','Box Plot']
    
    eda_select = st.sidebar.radio('Choose a Plot',eda_options)
    
    if eda_select == 'Pair plot':    
            
        
        
        
        df=pd.read_csv(path)
        
        hue = st.selectbox("Please specify a hue",df.columns)

        #st.write(df.columns)

        if st.sidebar.button('Visualize'):

            fig = plt.figure()
            fig=sns.pairplot(df,hue=hue)
            #st.plotly_chart(fig)
            st.pyplot(fig)
            return 
        
        
        
    if eda_select == 'Histogram':
        
        df=pd.read_csv(path)
        
        #st.write(df.columns)
        
        hue = st.selectbox("Please specify a hue",df.columns)
        
        a = st.selectbox("Please Specify a name attribute, the name will be used to label the data axis.",df.columns)
    
        
        if st.sidebar.button('Visualize'):
        
        
        
            fig = plt.figure()
            fig = sns.FacetGrid(df, hue=hue) \
               .map(sns.distplot, a)\
                .add_legend();

            st.pyplot(fig)
            
            return
            
            
    if eda_select == 'Box Plot':
        
        df=pd.read_csv(path)
        
        if st.sidebar.button('Visualize'):
            
            fig,ax=plt.subplots()
            ax=df.boxplot()
            
            st.pyplot(fig)
            
            return
        
        
    
    
        

        
def linear_reg():
    
    #ml_options=['Linear Regression','Classification']
    
    #ml_select = st.sidebar.radio('Choose a Method depending upon your data',ml_options)
    
    #if ml_select == 'Linear Regression':
        
    df=pd.read_csv(path)

    col_name = st.selectbox("Please Enter the name of column to predict",df.columns)


    #pred=int(st.text_input("Enter the value"))

    if st.sidebar.checkbox("Build"):

        df=pd.read_csv(path)
        df1=pd.read_csv(path1)


        y=df[col_name]
        y1=y

        df=df.drop([col_name], axis = 1)
        df1=df1.drop([col_name], axis = 1)

        x=df

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,random_state = 1)



        regressor = LinearRegression()
        regressor.fit(x_train,y_train)





        st.write("Model Build Successfully")

        st.write("Please play with the sliders to give input")

        l=[]

        for i in df1:

            l.append(st.sidebar.slider('{}'.format(i),min(df1[i]),max(df1[i]),min(df1[i])))




        rescale=['None','Min Max Scaler','Standard Scaler','Max Absolute Scaler','Robust Scaler']

        rs=st.sidebar.radio("How did you scaled your data?",rescale)

        if rs == 'Min Max Scaler':


            scaler = MinMaxScaler() # default min and max values are 0 and 1, respectively

            scaled_data = scaler.fit_transform(y1.values.reshape(-1,1))

            new_data=np.array(l)
            scaled_data = scaler.fit_transform(new_data.reshape(-1,1))

            if st.button("Predict"):



                y_pred = regressor.predict(scaled_data.T)
                np.set_printoptions(precision=3)
                orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(orig_data)))






        if rs == 'Standard Scaler':

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(y1.values.reshape(-1,1))
            new_data=np.array(l)
            scaled_data = scaler.fit_transform(new_data.reshape(-1,1))

            if st.button("predict"):



                y_pred = regressor.predict(scaled_data.T)
                np.set_printoptions(precision=3)
                orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(orig_data)))



        if rs == 'Max Absolute Scaler':

            scaler = MaxAbsScaler()
            scaled_data = scaler.fit_transform(y1.values.reshape(-1,1))
            new_data=np.array(l)
            scaled_data = scaler.fit_transform(new_data.reshape(-1,1))

            if st.button("predict"):



                y_pred = regressor.predict(scaled_data.T)
                np.set_printoptions(precision=3)
                orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(orig_data)))



        if rs == 'Robust Scaler':

            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(y1.values.reshape(-1,1))
            new_data=np.array(l)
            scaled_data = scaler.fit_transform(new_data.reshape(-1,1))

            if st.button("predict"):



                y_pred = regressor.predict(scaled_data.T)
                np.set_printoptions(precision=3)
                orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(orig_data)))




        if rs == 'None':






            if st.button("Predict."):


                y_pred = regressor.predict([l])
                np.set_printoptions(precision=3)
                #orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(y_pred)))


        st.sidebar.write("Happy with your model?")
        #st.sidebar.write("Download it ↓")

        if st.sidebar.button("Download it ↓"):
            Pkl_Filename = "Model.pkl"  

            with open(Pkl_Filename, 'wb') as file: 
                pickle.dump(regressor, file)

            st.sidebar.markdown(get_binary_file_downloader_html('Model.pkl', 'Model'), unsafe_allow_html=True)



                
def knn_classifier():
    df=pd.read_csv(path)

    col_name = st.selectbox("Please Enter the name of column to predict",df.columns)

    #st.write(df.columns)
    
    cv=st.sidebar.checkbox("Run with CV")
    
    if st.sidebar.checkbox("Analyze Optimal K & Build"):
        
        if cv==True:
            
        
            df=pd.read_csv(path)
            df1=pd.read_csv(path1)


            y=df1[col_name]
            y1=y

            df=df.drop([col_name], axis = 1)
            df1=df1.drop([col_name], axis = 1)

            x=df

            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,random_state = 1)

            mylist = list(range(0,50))

            neighbors = list(filter(lambda x : x%2!=0, mylist))

            cv_scores = []
            for k in neighbors:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn,x_train,y_train,cv=10, scoring='accuracy')
                cv_scores.append(scores.mean())

            MSE = [1-x for x in cv_scores]

            optimal_k = neighbors[MSE.index(min(MSE))]
            st.write('\n The optimal number of neighbors are: ', optimal_k)

            #fig, ax = plt.subplots()
            fig=plt.plot(neighbors,MSE)
            fig=plt.xlabel('Number of neighbors')
            fig=plt.ylabel('Misclassification Error')
            st.pyplot()

            knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
            knn_optimal.fit(x_train,y_train)
            pred = knn_optimal.predict(x_test)
            acc = accuracy_score(y_test,pred)*100
            st.write('The accuracy with KNN', optimal_k, 'we are getting: ', acc)

            st.write("Please play with the sliders to give input")

            l=[]

            for i in df1:

                l.append(st.sidebar.slider('{}'.format(i),min(df1[i]),max(df1[i]),min(df1[i])))


            if st.button("Predict."):
                y_pred = knn_optimal.predict([l])
                np.set_printoptions(precision=3)
                #orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(y_pred)))
                #st.write("Means Dead")
                
                
        if cv == False:
            df=pd.read_csv(path)
            df1=pd.read_csv(path1)


            y=df1[col_name]
            y1=y

            df=df.drop([col_name], axis = 1)
            df1=df1.drop([col_name], axis = 1)

            x=df
            
            x_1, x_test, y_1, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
            x_train, x_cv, y_train, y_cv = train_test_split(x_1,y_1,test_size=0.3,random_state=0)
            
            a=0
            
            for i in range(1,30,2):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(x_train,y_train)
                pred = knn.predict(x_cv)
                acc = accuracy_score(y_cv,pred,normalize=True)*float(100)
                if a<acc:
                    a=acc
                    k=i
                    
            st.info("Best accuracy we are getting is {} with {} neighbor".format(a,k))
                    
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            pred = knn.predict(x_test)
            acc = accuracy_score(y_test,pred, normalize=True)*float(100)
            st.info('Test data accuracy for k = {} is {}'.format(k,acc))
            
            st.info("Please play with the sliders to give input")

            l=[]

            for i in df1:

                l.append(st.sidebar.slider('{}'.format(i),min(df1[i]),max(df1[i]),min(df1[i])))


            if st.button("Predict."):
                y_pred = knn.predict([l])
                np.set_printoptions(precision=3)
                #orig_data = scaler.inverse_transform([y_pred])

                st.write("Your Predicted {} is {:.2f} :".format(col_name,float(y_pred)))
                
            
            
                

            
            





def ml_options():
    ml_option_list = ["Linear Regression","Classification"]
    
    cl_option_list = ["Knn Classifier","Naive Bayes"]
    
    ml_option_select = st.sidebar.radio("Select the machine learning method",ml_option_list)
    
    if ml_option_select == "Linear Regression":
        return ml_option_select
    
    if ml_option_select == "Classification":
        
        cl_option_select = st.sidebar.radio("Select the Classification method",cl_option_list)
        
        if cl_option_select == "Knn Classifier":
            return cl_option_select
        
        if cl_option_select == "Naive Bayes":
            st.write("You have not unlocked this achivement yet ")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("Just kidding This section will be build soon")
        
        
            
            
            
        
        
        
        





# Give main options


def main_option():
    
    try:
        
    
        option=('Missing Value Treatment', 'Outlier Treatment', 'Feature Scaling','Drop Columns')

        option_select = st.sidebar.radio('What would you like to do?',option)
        
        if option_select == "Drop Columns":

            
            df = pd.read_csv(path)
            
            st.write(df)
            
            col = col = df.columns.tolist()
            
            ch_col = st.multiselect("Please select Columns to drop",col)
            
            if st.button("Drop"):
                
                df = df.drop(ch_col, axis = 1)
                
                df.to_csv(path, index=False)
                df.to_csv(path1, index=False)
                
                st.write("Columns Dropped")
                
                st.write("Updated Dataframe")
                
                st.write(df)
                

        return option_select

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    
                        

def main():
    
        
    main_options=['Data Pre Processing','EDA','Machine Learning']

    ch = st.sidebar.selectbox('Choose what to do ', main_options)


    if ch == 'Data Pre Processing':
        df=file_upload()

        m_option = main_option()

        if m_option == 'Missing Value Treatment':

            df=mvt_options(df)


        elif m_option == 'Outlier Treatment':

            outlier_function()

        elif m_option == 'Feature Scaling':

            fso(df)


        data_export(df)


    if ch == 'EDA':

        df=file_upload()

        eda(df)
        
    if ch == 'Machine Learning':
    
        df = file_upload()
        
        ml_option = ml_options()
        
        if ml_option == "Linear Regression":
            
            linear_reg()
            
        if ml_option == "Knn Classifier":
            knn_classifier()
            
            
            




    

    #except Exception as e:
     #   st.write("Oops!", e.__class__, "occurred.")
      #  return df
    

main()
        
        

        
    
    
    
            
            
            
            
            
            
            
            
            
            

    
    
    
    
    
    
    

# BOT   


#st.sidebar.title("Need Help")

#get_text is a simple function to get user input from text_input
#def get_text():
 #   input_text = st.sidebar.text_input("You: ","So, what's in your mind")
  #  return input_text
    
    


    
#if st.sidebar.button('Initialize bot'):
 #   st.sidebar.title("Your bot is ready to talk to you")
  #  st.sidebar.title("""
   #         Help Bot  
    #    Just paste here the error you are getting 
     #       """)
    #user_input = get_text()
    #st.sidebar.write('Hello')*/

#if True:
 #   st.sidebar.text_area("Bot: Hello")
    
    
    
    
    
    
        
        
    
    
    
    
    
    
