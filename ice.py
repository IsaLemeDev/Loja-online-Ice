import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import numpy as np

df = pd.read_csv('/datasets/games.csv')
new_columns = []
for new_name in df:
    column_lowered = new_name.lower()
    new_columns.append(column_lowered)
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df['Total_sales'] = df['NA_sales'] + df['EU_sales'] + df['JP_sales'] + df['Other_sales']
df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')
counts = df['Year_of_Release'].value_counts().sort_index()
plt.figure(figsize=(10,6)) 
plt.plot(counts.index, counts.values, marker='o', linestyle='-') 
plt.title('Lançamentos por ano') 
plt.xlabel('Ano de lançamento') 
plt.ylabel('Quantidade de jogos') 
plt.grid(True) 
plt.show()

print(df['Platform'].value_counts())
print()
platform_sales = df.groupby('Platform')['Total_sales'].sum().sort_values(ascending=False)
print()
print(platform_sales.head(10))

plt.figure(figsize=(12,6)) 
plt.plot(platform_sales.index, platform_sales.values, marker='o', linestyle='-')
plt.title('Vendas por plataforma') 
plt.xlabel('Plataforma') 
plt.ylabel('Total de vendas') 
plt.xticks(rotation=45) 
plt.grid(True) 
plt.show()

top_platforms = platform_sales.head(5).index.tolist()
print("Top 5 plataformas:", top_platforms)
df_top_platforms = df[df['Platform'].isin(top_platforms)]
platform_year_sales = df_top_platforms.groupby(['Platform', 'Year_of_Release'])['Total_sales'].sum().reset_index()
print(platform_year_sales.head(60))

plt.figure(figsize=(12,6)) 
for platform in platform_year_sales['Platform'].unique(): 
    data = platform_year_sales[platform_year_sales['Platform'] == platform] 
    plt.plot(data['Year_of_Release'], data['Total_sales'], marker='o', label=platform) 
plt.title('Evolução das vendas por ano - Top 5 plataformas') 
plt.xlabel('Ano de lançamento') 
plt.ylabel('Total de vendas') 
plt.legend(title='Plataformas') 
plt.grid(True) 
plt.show()

df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')
recent_years = df[df['Year_of_Release'] >= 2013]
recent_platform_sales = recent_years.groupby('Platform')['Total_sales'].sum().sort_values(ascending=False)
print(recent_platform_sales.head(10))

top_10_platforms = platform_sales.head(10).index.tolist()
df_top_10 = df[df['Platform'].isin(top_10_platforms)]


plt.figure(figsize=(12, 10))
sns.boxplot(data=df_top_10, x='Platform', y='Total_sales')
plt.xticks(rotation=45)
plt.title('Vendas Globais - Top 10 Plataformas', fontsize=16)
plt.xlabel('Plataforma', fontsize=16)
plt.ylabel('Vendas Totais', fontsize=16)
plt.tight_layout()
plt.show()

venda_media = df.groupby('Platform')['Total_sales'].mean().sort_values(ascending=False)
print("Venda média em várias plataformas: ")
print(venda_media.head(10))
ps4_x360_data = df[df['Platform'].isin(['PS4', 'X360'])]

ps4_x360_clean = ps4_x360_data[
    (ps4_x360_data['Critic_Score'] != 'TBD') & 
    (ps4_x360_data['Critic_Score'] != 'tbd') &
    (ps4_x360_data['User_Score'] != 'TBD') &
    (ps4_x360_data['User_Score'] != 'tbd')
].copy()

ps4_x360_clean['Critic_Score'] = pd.to_numeric(ps4_x360_clean['Critic_Score'], errors='coerce')
ps4_x360_clean['User_Score'] = pd.to_numeric(ps4_x360_clean['User_Score'], errors='coerce')

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

ps4_clean = ps4_x360_clean[ps4_x360_clean['Platform'] == 'PS4']
axes[0,0].scatter(ps4_clean['Critic_Score'], ps4_clean['Total_sales'], alpha=0.6, color='blue')
axes[0,0].set_title('PS4: Critic Score vs Vendas Totais')
axes[0,0].set_xlabel('Critic Score')
axes[0,0].set_ylabel('Vendas Totais (milhões)')

axes[0,1].scatter(ps4_clean['User_Score'], ps4_clean['Total_sales'], alpha=0.6, color='blue')
axes[0,1].set_title('PS4: User Score vs Vendas Totais')
axes[0,1].set_xlabel('User Score')
axes[0,1].set_ylabel('Vendas Totais (milhões)')

x360_clean = ps4_x360_clean[ps4_x360_clean['Platform'] == 'X360']
axes[1,0].scatter(x360_clean['Critic_Score'], x360_clean['Total_sales'], alpha=0.6, color='green')
axes[1,0].set_title('X360: Critic Score vs Vendas Totais')
axes[1,0].set_xlabel('Critic Score')
axes[1,0].set_ylabel('Vendas Totais (milhões)')

axes[1,1].scatter(x360_clean['User_Score'], x360_clean['Total_sales'], alpha=0.6, color='green')
axes[1,1].set_title('X360: User Score vs Vendas Totais')
axes[1,1].set_xlabel('User Score')
axes[1,1].set_ylabel('Vendas Totais (milhões)')

plt.tight_layout()
plt.show()

critic_corr = ps4_clean['Critic_Score'].corr(ps4_clean['Total_sales'])
user_corr = ps4_clean['User_Score'].corr(ps4_clean['Total_sales'])

print(f'Correlação Critic Score vs Vendas (PS4): {critic_corr:.2f}')
print(f'Correlação User Score vs Vendas (PS4): {user_corr:.2f}')

na_platform_sales = df.groupby('Platform')['NA_sales'].sum().sort_values(ascending=False)
print("Top 5 plataformas na América do Norte:")
print(na_platform_sales.head(5))
print()
eu_platform_sales = df.groupby('Platform')['EU_sales'].sum().sort_values(ascending=False)
print("Top 5 plataformas na Europa:")
print(eu_platform_sales.head(5))
print()

jp_platform_sales = df.groupby('Platform')['JP_sales'].sum().sort_values(ascending=False)
print("Top 5 plataformas no Japão: ")
print(jp_platform_sales.head(5))

na_genre_sales = df.groupby('Genre')['NA_sales'].sum().sort_values(ascending=False)
print("Top 5 generos na América do Norte:")
print(na_genre_sales.head(5))
print()
eu_genre_sales = df.groupby('Genre')['EU_sales'].sum().sort_values(ascending=False)
print("Top 5 generos na Europa:")
print(eu_genre_sales.head(5))
print()
jp_genre_sales = df.groupby('Genre')['JP_sales'].sum().sort_values(ascending=False)
print("Top 5 generos no Japão:")
print(jp_genre_sales.head(5))

na_rating_sales = df.groupby('Rating')['NA_sales'].sum().sort_values(ascending=False)
print("Top 5 classificações na América do Norte:")
print(na_rating_sales.head(5))
print()
eu_rating_sales = df.groupby('Rating')['EU_sales'].sum().sort_values(ascending=False)
print("Top 5 classificações na Europa:")
print(eu_rating_sales.head(5))
print()
jp_rating_sales = df.groupby('Rating')['JP_sales'].sum().sort_values(ascending=False)
print("Top 5 classificações no Japão:")
print(jp_rating_sales.head(5))

xbox_games = df[df['Platform'] == 'XOne'] 
pc_games = df[df['Platform'] == 'PC']
xbox_scores_clean = pd.to_numeric(xbox_games['User_Score'], errors='coerce').dropna()
pc_scores_clean = pd.to_numeric(pc_games['User_Score'], errors='coerce').dropna()

print('*** Teste de hipótese: Xbox One x PC ***')
print('(Hipótese Nula) H0: A classificação média dos usuários é igual entre Xbox One e PC')
print('(Hipótese Alternativa): A classificação média dos usuários é diferente entre Xbox One e PC')
print()

alpha = 0.05
print(f"Nível de significância (α): {alpha}")
print()

xbox_variance = xbox_scores_clean.var()
pc_variance = pc_scores_clean.var()
print(f"Variância Xbox One: {xbox_variance:.4f}")
print(f"Variância PC: {pc_variance:.4f}")

ratio = max(xbox_variance, pc_variance) / min(xbox_variance, pc_variance)
print(f"Razão das variâncias: {ratio:.4f}")

equal_variances = ratio <= 4
print(f"Variâncias consideradas iguais? {equal_variances}")
print()

results = st.ttest_ind(
    xbox_scores_clean, 
    pc_scores_clean,
    equal_var=equal_variances
)

print(f"Estatística t: {results.statistic:.4f}")
print(f"Valor-p: {results.pvalue:.4f}")
print()

if results.pvalue < alpha:
    print("Rejeitamos a hipótese nula")
    print("Conclusão: Há diferença significativa entre as classificações médias do Xbox One e PC")
else:
    print("Não podemos rejeitar a hipótese nula")
    print("Conclusão: Não há evidência suficiente de diferença entre as classificações médias do Xbox One e PC")

action_games = df[df['Genre'] == 'Action']
sports_games = df[df['Genre'] == 'Sports']
action_scores_clean = pd.to_numeric(action_games['User_Score'], errors='coerce').dropna()
sports_scores_clean = pd.to_numeric(sports_games['User_Score'], errors='coerce').dropna()

print('*** Teste de hipótese: Action x Sports***')
print('(Hipótese Nula) H0: A classificação média dos usuários por gênero é igual entre Action e Sports')
print('(Hipótese Alternativa) H1: A classificação média dos usuários por gênero é diferente entre Action e Sports')
print()

alpha = 0.05
print(f"Nível de significância (α): {alpha}")
print()

action_variance = action_scores_clean.var()
sports_variance = sports_scores_clean.var()
print(f"Variância Actione: {action_variance:.4f}")
print(f"Variância Sports: {sports_variance:.4f}")

ratio = max(action_variance, sports_variance) / min(action_variance, sports_variance)
print(f"Razão das variâncias: {ratio:.4f}")

equal_variances = ratio <= 4
print(f"Variâncias consideradas iguais? {equal_variances}")
print()

results = st.ttest_ind(
    action_scores_clean, 
    sports_scores_clean,
    equal_var=equal_variances
)

print(f"Estatística t: {results.statistic:.4f}")
print(f"Valor-p: {results.pvalue:.4f}")
print()


if results.pvalue < alpha:
    print("Rejeitamos a hipótese nula")
    print('Conclusão: Há diferença significativa entre as classificações médias do gênero Action e Sports.')
else:

    print("Não podemos rejeitar a hipótese nula")
    print('Conclusão: Não há evidência suficiente de diferença entre as classificações médias do gênero Action e Sports.')

# Conclusão do projeto: Entre os dois gêneros Sports e Action, a hipótese nula não tem diferença significativa. Já a hipótese alternativa tem diferença significativa. A diferença entre os gêneros é de 0.09, sendo maior que o alfa escolhido, podemos dizer que existe diferença significativa entre as médias das avaliações dos dois gêneros. Então não podemos desconsiderar a hipótese nula. Entre as duas Plataformas, Xbox One e PS, a hipótese nula não tem diferença significativa. Já a hipótese alternativa tem diferença significativa. A diferença entre os jogos é de 0.54, sendo maior que o alfa escolhido, podemos dizer que existe diferença significativa entre as médias das avaliações dos dois gêneros. Então não podemos desconsiderar a hipótese nula.