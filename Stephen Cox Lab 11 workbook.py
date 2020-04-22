import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

import pandas as pd
import sqlite3

conn = sqlite3.connect("DCA.db") 

cur = conn.cursor()   
cur.execute("CREATE TABLE DCAparams (wellID INTEGER,  qi REAL, Di REAL, b REAL)")
conn.commit()

dfLength = 24

for i in range(1,18):
    wellID = i
    fileName = 'DCAwells_Solved\DCA_Well ' + str(wellID) + '.xlsx'  
    xl = pd.ExcelFile(fileName)

    df1 = xl.parse('DCARegression')
    rateDF = pd.DataFrame({'wellID':wellID*np.ones(dfLength,dtype=int), 'time':range(1,dfLength+1),'rate':df1.iloc[8:32,1].values})
    
    rateDF['Cum'] = rateDF['rate'].cumsum()
    
    qi = df1.iloc[2,3]
    Di = df1.iloc[3,3]
    b  = df1.iloc[4,3]
    
    cur.execute(f"INSERT INTO DCAparams VALUES ({wellID}, {qi}, {Di}, {b})")
    conn.commit()
    
    t = np.arange(1,dfLength+1)
    Di = Di/12
    
    if b==1:
        q = 30.4375*qi/((1 + Di*t)) 
        Np = 30.4375*((qi/Di)*np.log(1+(Di*t)))
    elif 1>b>0 or b>1:
        q = 30.4375*qi/((1 + b*Di*t)**(1/b))
        Np = 30.4375*(qi/(Di*(1-b)))*(1-(1/(1+(b*Di*t))**((1-b)/b)))
    elif b==0:
        q = qi*np.exp(-Di*t)   
        Np = 30.4375*(qi-q)/Di 
        q = 30.4375*q      
        
    rateDF['q_model'] = q
    rateDF['Cum_model'] = Np
    
    rateDF.to_sql("Rates", conn, if_exists="append", index = False)

    error_q = rateDF['rate'].values - q
    SSE_q = np.dot(error_q, error_q)
    
    errorNp = rateDF['Cum'].values - Np
    SSE_Np = np.dot(errorNp, errorNp)

    prodDF = pd.read_sql_query(f"SELECT * FROM Rates WHERE wellID={wellID};", conn)
    dcaDF = pd.read_sql_query(f"SELECT * FROM DCAparams WHERE wellID={wellID};", conn)

    titleFontSize = 18
    axisLabelFontSize = 15
    axisNumFontSize = 13

    currFig = plt.figure(figsize=(7,5), dpi=100)

    axes = currFig.add_axes([0.15, 0.15, 0.7, 0.7])

    axes.plot(prodDF['time'], prodDF['Cum']/1000, color="red", ls='None', marker='o', markersize=4,label = 'well '+ str(wellID) + 'actual' )
    axes.plot(prodDF['time'], prodDF['Cum_model']/1000, color="blue", lw=1, ls='-',label = 'well '+ str(wellID) +'predicted' )
    axes.legend(loc=4)
    axes.set_title('Cumulative Production vs Time', fontsize=titleFontSize, fontweight='bold')
    axes.set_xlabel('Time, Months', fontsize=axisLabelFontSize, fontweight='bold') 
    axes.set_ylabel('Cumulative Production, Mbbls', fontsize=axisLabelFontSize, fontweight='bold')
    ya = axes.get_yaxis()
    ya.set_major_locator(MaxNLocator(integer=True))
    
    axes.set_xlim([0, 25])
    xticks = range(0,30,5) 
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticks, fontsize=axisNumFontSize); 
    

    currFig.savefig('well'+ str(i) +'_production.png', dpi=600)


conn.close()

