import itertools

df = pd.DataFrame(list(itertools.product([0, 1, 2], repeat=9)), columns=['TopLeft', 'TopMid', 'TopRight', 'MidLeft', 'MidMid', 'MidRight', 'BotLeft', 'BotMid', 'BotRight'])
l1 = df.columns.tolist().copy()
df['num1'] = df.apply(lambda x: (x[l1].values==1).sum(), axis=1)
df['num2'] = df.apply(lambda x: (x[l1].values==2).sum(), axis=1)
df['diff'] = abs(df['num1']-df['num2'])
df = df[df['diff']<=1]
df = df.drop(['num1', 'num2', 'diff'], axis=1)

df.to_csv('all_states.csv')
