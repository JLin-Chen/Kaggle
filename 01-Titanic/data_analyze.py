# use % matplotlib inline in jupyter notebook,plt.show() in Pycharm instead
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat([train, test], ignore_index=True)

print(train.head())  #查看总体数据
print(train.info())  #查看数据情况

# 进行统计学分析
# 存活情况
print(train['Survived'].value_counts)
# 性别特征
plt.figure(1)
g1 = sns.barplot(x=train['Sex'], y=train['Survived']).set_title('sex feature')
# 阶级特征
plt.figure(2)
g2 = sns.barplot(x=train['Pclass'], y=train['Survived']).set_title('class feature')
# 远亲关系特征
plt.figure(3)
sns.barplot(x=train['SibSp'], y=train['Survived'])
# 近亲关系特征
plt.figure(4)
sns.barplot(x=train['Parch'], y=train['Survived'])
# 年龄特征

facet = sns.FacetGrid(train, hue="Survived", aspect=2)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.figure(5)
plt.xlabel('Age')
plt.ylabel('density')
plt.show()
# 登港港口特征
plt.figure(6)
sns.countplot('Embarked', hue='Survived', data=train)
# 称谓特征
plt.figure(7)
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data)
# 家庭人数特征
plt.figure(8)
all_data['FamilySize'] = all_data['SibSp']+all_data['Parch']+1
sns.barplot(x='FamilySize', y='Survived', data=all_data)
# 定义家庭等级&特征
def Fam_lable(s):
    if(s>=2)&(s<=4):
        return 2
    elif ((s>4)&(s<=7)|(s==1)):
        return 1
    elif (s>7):
        return 0
plt.figure(9)
all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_lable)
sns.barplot(x=all_data['FamilyLabel'], y=all_data['Survived'])
# 甲板特征
plt.figure(10)
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data)
# 共票号特征
plt.figure(11)
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])
sns.barplot(x="TicketGroup", y="Survived", data=all_data)
# 票号的等级分类&特征
def Ticket_Label(s):
    if(s>=2)&(s<=4):
        return 2
    elif((s>4)&(s<=8))|(s==1):
        return 1
    elif(s>8):
        return 0
all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data)

plt.show()  #用于显示文中出现的所有图例
