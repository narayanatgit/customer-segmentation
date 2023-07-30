from flask import Flask, render_template, url_for, jsonify,send_file
import pandas as pd
import datetime as dt
import scipy.stats as stats
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math
import numpy as np
from sklearn.metrics import silhouette_score
import math
import io
import base64
import numpy as np
app = Flask(__name__)

df = pd.read_csv('53online.csv',encoding= 'unicode_escape')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/table')
def table():    
    df_head = df.head()
    df_tail = df.tail()
    table_html = df_head.to_html(classes='table table-striped')   
    tail_html = df_tail.to_html(classes='table table-striped')   
    shape=df.shape
    describe=df.describe()
    describe =describe.to_html()  
    missing_counts = df.isnull().sum()
    missing_counts_df = pd.DataFrame({'column': missing_counts.index, 'missing_count': missing_counts.values})
    missing_counts_html = missing_counts_df.to_html(index=False)
    table={'head':table_html,'tail':tail_html,'shape':shape,'des':describe,'info':missing_counts_html}
    return jsonify(table)
@app.route('/pre')
def pre(): 
    totalrows=len(df)
    duplicateRows=df[df.duplicated()]
    dup=df.drop_duplicates()
    nodup=len(df)-len(dup)
    filter1=df['Quantity']>0
    flter2=df['CustomerID'].isnull()!=True
    filter3=df['Description'].isnull()!=True
    filter4=df['UnitPrice']>0
    global drop
    drop=df.where(filter1 & flter2 & filter3&filter4).dropna()
    df_new=drop[drop['UnitPrice']==0]
    nocus=len(drop['CustomerID'].unique())
    afterpre=len(drop)
    noditems=len(drop['Description'].unique())
    pre={'total':totalrows,'dup':nodup,'nocus':nocus,'afterpre':afterpre,'noofitems':noditems}
    return jsonify(pre)
@app.route('/rfm')
def rfm(): 
    origin=pd.DataFrame(drop)
    oi=origin
    cus=origin.groupby('CustomerID')
    mv=origin.groupby('CustomerID').apply(lambda x:((x['Quantity']*x['UnitPrice']).sum()))
    origin['InvoiceDate']=pd.to_datetime(origin['InvoiceDate'],dayfirst=True)
    maxamt=mv.max()
    lastdate=pd.to_datetime('11/12/2011',dayfirst=True)
    maxdate=origin['InvoiceDate'].max()
    global fre
    fre=origin.groupby('CustomerID').agg({'InvoiceDate': lambda x:(lastdate - x.max()).days,'InvoiceNo' :lambda x: len(x)})
    fre['price']=mv
    fre['Recency']=pd.qcut(fre['InvoiceDate'], q=5,labels=[5,4,3,2,1])
    fre['frequency']=pd.qcut(fre['InvoiceNo'], q=5, labels=[1,2,3,4,5])
    fre['Montary'] = pd.qcut(fre['price'], q=5, labels=[1,2,3,4,5])
    fre['RFMSCORE']=fre['Recency'].astype(str)+fre['frequency'].astype(str)+fre['Montary'].astype(str)
    fre['Recency']=fre['Recency'].astype(int)
    fre['frequency']=fre['frequency'].astype(int)
    fre['Montary']=fre['Montary'].astype(int)
    fre['sum']=fre['Recency']+fre['frequency']+fre['Montary']
    fre_html=fre.to_html(classes='table table-striped')
    rfm={'fre':fre_html,'max':maxamt,'refer':lastdate,'maxd':maxdate}
    return jsonify(rfm)
@app.route('/kmeans')
def kmeans(): 
    
   
    skewness1 = stats.skew(fre['InvoiceDate'])
    skewness2 = stats.skew(fre['InvoiceNo'])
    skewness3= stats.skew(fre['price'])
    global rty
    rty=fre
    op=fre[['InvoiceDate','InvoiceNo','price']]

    data1,_=stats.boxcox(op['InvoiceDate'])
    data2,_=stats.boxcox(op['InvoiceNo'])
    data3,_=stats.boxcox(op['price'])
    op = op.assign(InvoiceDate=data1)
    op = op.assign(InvoiceNo=data2)
    op = op.assign(price=data3)
    skewness1r = stats.skew(op['InvoiceDate'])
    skewness2r = stats.skew(op['InvoiceNo'])
    skewness3r= stats.skew(op['price'])
    scaler = StandardScaler()
    global datast
    datast =pd.DataFrame(scaler.fit_transform(op),columns=op.columns,index=op.index)
    
    desi=datast.describe().to_html(classes='w3-table-all w3-centered')
    ori=len(datast)
    q1re, q3re = np.percentile(datast['InvoiceDate'], [25, 75])
    iqrre = q3re - q1re
    lower_boundre = q1re - 1.5* iqrre
    upper_boundre = q3re + 1.5 * iqrre
    filter34=datast['InvoiceDate'].apply(lambda x: x>=lower_boundre and x<=upper_boundre)

    q1fe, q3fe = np.percentile(datast['InvoiceNo'], [25, 75])
    iqrfe = q3fe - q1fe
    lower_boundfe = q1fe - 1.5* iqrfe
    upper_boundfe = q3fe + 1.5 * iqrfe
    filter35=datast['InvoiceNo'].apply(lambda x: x>=lower_boundfe and x<=upper_boundfe)

    q1me, q3me = np.percentile(datast['price'], [25, 75])
    iqrme = q3me - q1me
    lower_boundme = q1me - 1.5* iqrme
    upper_boundme = q3me + 1.5 * iqrme
    filter36=datast['price'].apply(lambda x: x>=lower_boundme and x<=upper_boundme)
    check3=datast.where(filter35&filter36&filter34)

    datast=datast.where(filter35&filter36&filter34).dropna()
    after=len(datast)
    totalo=ori-len(datast)
    datst_html=datast.to_html()
    kmean={'skew1':skewness1,'skew2':skewness2,'skew3':skewness3,'skew1r':skewness1r,'skew2r':skewness2r,'skew3r':skewness3r,
           
           'std':desi,'before':ori,'after':datst_html,'total':totalo}
    
    return jsonify(kmean)
@app.route('/cluster')
def cluster():
    start_time = time.time()
    kmeans = KMeans(n_clusters=3,init='k-means++',random_state=0 )
    kmeans.fit(datast[['price','InvoiceNo']])
    global no_of_ite1
    no_of_ite1=kmeans.n_iter_
    print(no_of_ite1)
    centroids8 = kmeans.cluster_centers_

    
    labels=kmeans.labels_
    global ktime
    ktime= time.time()-start_time
    datast3=datast[['price','InvoiceNo']]
    silhouette_vals31 = silhouette_samples(datast3,labels )
    global silhouette_avgr31
    silhouette_avgr31 = silhouette_score(datast3, labels)
    datast['cul1']=kmeans.labels_
    kmeans.fit(datast[['price','InvoiceDate']])
    centroids9 = kmeans.cluster_centers_
    
    datast['cul2']=kmeans.labels_
    datast5=datast[['InvoiceNo','InvoiceDate','price']]
    kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0 )
    kmeans.fit(datast5[['price','InvoiceNo']])
    centroids1 = kmeans.cluster_centers_
    datast['Recency']=fre['Recency']
    datast['frequency']=fre['frequency']
    datast['Monetary']=fre['Montary']
    data_html=datast.to_html(classes='table table-striped')
    datast52=datast[['InvoiceNo','price']]
    labels=kmeans.labels_
    silhouette_avgr52 = silhouette_score(datast52,labels)
    silhouette_vals52 = silhouette_samples(datast52,labels)
    datast5['cul51']=kmeans.labels_
    kmeans.fit(datast5[['price','InvoiceDate']])
    centroids2 = kmeans.cluster_centers_
    
    datast5['cul52']=kmeans.labels_
    cl={'sil1':silhouette_avgr31,'sil2':silhouette_avgr52,'data1':data_html}
    return jsonify(cl)
@app.route('/rm')
def rm():
    x=datast[['InvoiceNo','price']]
    k=3
    ctr=0
    max_iter=100

    start_time = time.time()
    x = x.sort_values(by=['price','InvoiceNo'])

    yq =x.reset_index().rename(columns={'index': 'customerID'})
    x =x.reset_index(drop=True)

    num_splits = 3
    split_data = np.array_split(x, num_splits)

    centroids = np.zeros((num_splits, x.shape[1]))

    for i, split in enumerate(split_data):
        centroids[i] = np.median(split, axis=0)
        
    from scipy.spatial.distance import cdist
    f=0
    for i in range(max_iter):
            ctr=ctr+1
            # Assign each data point to the nearest centroid
            distances = cdist(x, centroids)
            labels14 = np.argmin(distances, axis=1)

            # Compute the median of the data points assigned to each centroid
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                new_centroids[j] = np.median(x[labels14 == j], axis=0)

            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=0):
                f=ctr
                break

            centroids = new_centroids
    global rmtime
    rmtime = time.time()-start_time
    global iterationrm
    iterationrm=f
  

    x['index1']=x.index
    x['clusters']=labels14
    yq['index']=yq.index


    merge_df=pd.merge(x,yq,left_on='index1',right_on='index')
    merge_df=merge_df[['InvoiceNo_x','price_x','clusters','CustomerID']]
    merge_df=merge_df.set_index('CustomerID')

    merge_df1=merge_df[['InvoiceNo_x','price_x']]
    global silhouette_avgrrm1
    silhouette_avgrrm1 = silhouette_score(merge_df, labels14)
    silhouette_valsrm1 = silhouette_samples(merge_df,labels14)
    merge_df['Recency']=rty['Recency']
    merge_df['frequency']=rty['frequency']
    merge_df['Monetary']=rty['Montary']
    merge_df_html=merge_df.to_html(classes='table table-striped')
    x=datast[['InvoiceDate','price']]
    k=3
    ctr=0
    max_iter=100


    x = x.sort_values(by=['price','InvoiceDate'])

    yq =x.reset_index().rename(columns={'index': 'customerID'})
    x =x.reset_index(drop=True)

    num_splits = 3
    split_data = np.array_split(x, num_splits)

    centroids = np.zeros((num_splits, x.shape[1]))

    for i, split in enumerate(split_data):
        centroids[i] = np.median(split, axis=0)
        
    from scipy.spatial.distance import cdist
    for i in range(max_iter):
            ctr=ctr+1
            # Assign each data point to the nearest centroid
            distances = cdist(x, centroids)
            labels14 = np.argmin(distances, axis=1)

            # Compute the median of the data points assigned to each centroid
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                new_centroids[j] = np.median(x[labels14 == j], axis=0)

            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=0):
                f=ctr
                break

            centroids = new_centroids



    x['index1']=x.index
    x['clusters']=labels14
    yq['index']=yq.index


    merge_df=pd.merge(x,yq,left_on='index1',right_on='index')
    merge_df=merge_df[['InvoiceDate_x','price_x','clusters','CustomerID']]
    merge_df=merge_df.set_index('CustomerID')

    merge_df1=merge_df[['InvoiceDate_x','price_x']]
    merge_df['Recency']=rty['Recency']
    merge_df['frequency']=rty['frequency']
    merge_df['Monetary']=rty['Montary']
    merge_dfhtml1=merge_df.to_html(classes='table table-striped')
    rm={"rm1":merge_df_html,'rm2':merge_dfhtml1}
    return jsonify(rm)
@app.route('/ana')
def ana():
    optmi=pd.DataFrame({
    'Algorithm':["K-means","RM K-means"],
    'Iterations':[no_of_ite1,iterationrm],
    'Time':[ktime,rmtime],
    'Cluster compactness':[silhouette_avgr31,silhouette_avgrrm1]
    })
    ryu=optmi.to_html()
    ana={'ana':ryu,'ktime':ktime,'rmtime':rmtime,'sil1':silhouette_avgr31,'sil2':silhouette_avgrrm1,'it1':no_of_ite1,'it2':iterationrm}
    return jsonify(ana)
@app.route('/rep')
def rep():
    f="Result"
    return f
if __name__ == '__main__':
    app.run(debug=True)