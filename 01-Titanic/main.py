# use % matplotlib inline in jupyter notebook,plt.show() in Pycharm instead
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
PassengerId = test['PassengerId']
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
# plt.show()
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

# plt.show()  #用于显示文中出现的所有图例


# 缺失值填充 年龄
# Age缺失量为263，缺失量较大，用Sex, Title, Pclass三个特征构建随机森林模型，填充年龄缺失值。
from sklearn.ensemble import RandomForestRegressor
age_df = all_data[['Age', 'Pclass', 'Sex', 'Title']]
age_df = pd.get_dummies(age_df)  # 用pandas实现one hot encode
# known_age = age_df[age_df.Age.notnull()].as_matrix()
# unknown_age = age_df[age_df.Age.isnull()].as_matrix()
known_age = age_df[age_df.Age.notnull()].iloc[:,:].values
unknown_age = age_df[age_df.Age.isnull()].iloc[:,:].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges

# Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。
print(all_data['Embarked'].isnull())
print(all_data.groupby(by=["Pclass", "Embarked"]).Fare.median())
all_data['Embarked'] = all_data['Embarked'].fillna('C')

# Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。
print(all_data['Fare'].isnull())
fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

# 同组识别
# 把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
Female_Child_Group = all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12)|(all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
print(Female_Child)
sns.barplot(x=Female_Child.index, y=Female_Child['GroupCount']).set_xlabel('AverageSurvived')
# plt.show()
# 绝大部分成年男性组的平均存活率也为1或0。
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
print(Male_Adult)

# 因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，推测处于遇难组的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高。
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
# print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
# print(Survived_List)

# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'

# 选取特征，转换为数值变量，划分训练集和测试集。
all_data=pd.concat([train, test])
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.iloc[:,1:].values
y = train.iloc[:,0].values


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest


# 参数优化
pipe=Pipeline([('select',SelectKBest(k=20)),
               ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)),
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)

# 训练模型
from sklearn.pipeline import make_pipeline
select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state = 10, warm_start = True,
                                  n_estimators = 26,
                                  max_depth = 6,
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

# 交叉验证
from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

# 预测
predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission1.csv", index=False)
