import pandas as pd


# Reading the Sheet
df_read     = ['tempo','p','q','r']
df_pqr         = pd.read_excel('Anexo_A.xls', sheet_name = 'Sheet14', usecols = df_read, skiprows = 5)
#print(df_pqr) 
    
df_read1 = ['teta0', 'fi0', 'psi0']
df_ang = pd.read_excel('Anexo_A.xls', sheet_name = 'Sheet14', usecols = df_read1, skiprows = 1, nrows=1)
#print(df_ang)
