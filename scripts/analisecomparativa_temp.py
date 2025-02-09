# pcg_df = pcg_df.drop(index=[491, 503])


# len(pcg_df)


# # # Análise comparativa


# # Load Excel file
# df_excel = pd.read_excel("Frederikke_Annotations.xlsx")  # Change file name accordingly

# # Ensure column names match
# df_excel.rename(columns={"Patientid": "ID", "Valve": "Auscultation Point"}, inplace=True)

# # Convert ID in both dataframes to the same format (remove 'id' prefix if needed)
# df_excel["ID"] = df_excel["ID"].astype(str)
# pcg_df["ID"] = pcg_df["ID"].str.replace("id", "")  # Remove "id" if needed

# # Merge both dataframes based on ID & Auscultation Point
# merged_df = pcg_df.merge(df_excel, on=["ID", "Auscultation Point"], how="inner")

# # Display merged data
# print(merged_df)


# # In[ ]:


# merged_df['error_S1S2_start'] = abs(merged_df['S1S2']-1000*merged_df['avg_s1s2_intervals_start'])
# merged_df['error_S1S2_end'] = abs(merged_df['S1S2']-1000*merged_df['avg_s1s2_intervals_end'])
# merged_df['error_S1S2_mid'] = abs(merged_df['S1S2']-1000*merged_df['avg_s1s2_intervals_mid'])


# # In[ ]:


# final_df = pd.DataFrame(columns=['ID', 'Auscultation Point','True S1S2', 'UNET S1S2_start', 'S1S2_start error', 'UNET S1S2_end', 'S1S2_end error','S1S2_mid error','UNET S1S2_mid','Status_of_EF'])

# final_df['Auscultation Point'] = merged_df['Auscultation Point']
# final_df['ID'] = merged_df['ID']
# final_df['Status_of_EF']=merged_df['Status_of_EF']
# final_df['True S1S2'] = merged_df['S1S2']
# final_df['S1S2_start error'] = merged_df['error_S1S2_start']
# final_df['UNET S1S2_start'] = 1000*merged_df['avg_s1s2_intervals_start']
# final_df['S1S2_end error'] = merged_df['error_S1S2_end']
# final_df['UNET S1S2_end'] = 1000*merged_df['avg_s1s2_intervals_end']
# final_df['S1S2_mid error'] = merged_df['error_S1S2_mid']
# final_df['UNET S1S2_mid'] = 1000*merged_df['avg_s1s2_intervals_mid']


# # In[ ]:


# file_path = os.path.join('/content/drive/MyDrive/Tese/Daniel_PCG_UNET/Resultados Mariana', "PCG_S1S2.csv")
# final_df.to_csv(file_path, index=False)


# # In[ ]:


# error_stats_EF = df_S1S2.groupby("Status_of_EF")[["S1S2_start error", "S1S2_end error", "S1S2_mid error"]].agg(["mean", "median", "std"])
# print(error_stats_EF)


# # In[ ]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# sns.boxplot(x="Status_of_EF", y="S1S2_mid error", data=df_S1S2)
# plt.title("S1S2 Mid Error Distribution Across EF Status Groups")
# plt.show()


# # In[ ]:


# from scipy.stats import f_oneway

# groups = [df_S1S2[df_S1S2["Status_of_EF"] == ef]["S1S2_mid error"] for ef in df_S1S2["Status_of_EF"].unique()]
# stat, p = f_oneway(*groups)
# print(f"ANOVA p-value: {p}")


# # In[ ]:


# from scipy.stats import kruskal
# stat, p = kruskal(*groups)
# print(f"Kruskal-Wallis p-value: {p}")


# # In[ ]:


# from scipy.stats import spearmanr

# corr, p = spearmanr(df_S1S2["Status_of_EF"], df_S1S2["S1S2_mid error"])
# print(f"Spearman Correlation: {corr}, p-value: {p}")


# # In[ ]:


# import statsmodels.api as sm

# X = df_S1S2[["S1S2_start error", "S1S2_end error", "S1S2_mid error"]]
# y = df_S1S2["Status_of_EF"]
# X = sm.add_constant(X)  # Add intercept

# model = sm.Logit(y, X)
# result = model.fit()
# print(result.summary())