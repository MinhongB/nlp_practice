# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 백민홍
- 리뷰어 : Donggyu Kim

---------------------------------------------
## **PRT(PeerReviewTemplate)**
### Mission Object
1. 워드임베딩의 most_similar() 메소드 결과가 의미상 바르게 나왔다.
2. 타당한 방법론을 통해 중복이 잘 제거되고 개념축을 의미적으로 잘 대표하는 단어 셋이 만들어졌다.
3. 전체 영화 장르별로 예술/일반 영화에 대한 편향성 WEAT score가 상식에 부합하는 수치로 얻어졌으며 이를 잘 시각화하였다.

### **[x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
### **[x] 주석을 보고 작성자의 코드가 이해되었나요?**
### **[x] 코드가 에러를 유발할 가능성이 있나요?**
### **[x] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
### **[x] 코드가 간결한가요?**

## Evidences
### Section 1

#### 1st object
most_similar 
```python
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
print(model.wv.most_similar(positive=['사랑__nng']))
print('\n')
print(model.wv.most_similar(positive=['연극__nng']))
print('\n')
print(model.wv.most_similar(positive=['영화__nng']))
print('\n')
print(model.wv.most_similar(positive=['영화제__nng']))
print('\n')
print(model.wv.most_similar(positive=['가족__nng']))
```

result
```
[('만남__nng', 0.6491414308547974), ('키스__nng', 0.6291918158531189), ('우정__nng', 0.6056089401245117), ('고민__nng', 0.5845113396644592), ('경험__nng', 0.5794662237167358), ('기억__nng', 0.565150260925293), ('마음__nng', 0.5612759590148926), ('행복__nng', 0.5575109720230103), ('추억__nng', 0.5568584203720093), ('대면__nng', 0.5548210144042969)]


[('공연__nng', 0.7515076398849487), ('무용__nng', 0.7504252791404724), ('할리우드__nnp', 0.7349514961242676), ('뮤지컬__nng', 0.7233029007911682), ('미술__nng', 0.722792387008667), ('음악__nng', 0.712272584438324), ('오페라__nng', 0.7110645174980164), ('로드__nng', 0.7072425484657288), ('뮤직__nng', 0.6993142366409302), ('곡__nng', 0.6990588307380676)]
```

Everything is okay.
That is acceptable answers.

#### 2nd object

Code
```python
total_counts = Counter('')
temp = []
for i in range(len(genre)):
    word_counts = Counter(genre[i])
    total_counts = total_counts + word_counts
    temp.append(word_counts)
temp2 =[]
for i in range(len(genre)):
    word_ratios = {word: count / total_counts[word] if word in total_counts else 1 for word, count in temp[i].items() if count > 30}
    if len(word_ratios)<15 :
        print(genre_name[i]+'은 너무 짧습니다. 기준점을 바꿔서 다시 추출합니다.')
        word_ratios = {word: count / total_counts[word] if word in total_counts else 1 for word, count in temp[i].items() if count > 15}
    if len(word_ratios)<15 :
        print(genre_name[i]+'은 너무 짧습니다. 기준점을 바꿔서 다시 추출합니다.')
        word_ratios = {word: count / total_counts[word] if word in total_counts else 1 for word, count in temp[i].items() if count > 5}
    sorted_word_ratios = sorted(word_ratios.items(), key=lambda x: x[1], reverse=True)
    temp2.append(sorted_word_ratios)

for tt in range(len(temp2)) :
    print('\n\n'+genre_name[tt], end=':')
    for i in range(min(30,len(temp2[tt]))) :
#         print(str(temp2[tt][i]), end=', ')
        print(str(temp2[tt][i][0]), end=', ')
    
```

Result
```

SF:우주선__nng, 외계__nng, 인류__nng, 생명체__nng, 행성__nng, 로봇__nng, 지구__nng, 과학자__nng, 시스템__nng, 박사__nng, 우주__nng, 미래__nng, 연구__nng, 능력__nng, 실험__nng, 음모__nng, 공격__nng, 개발__nng, 거대__nng, 인간__nng, 정부__nng, 살아남__vv, 임무__nng, 요원__nng, 위협__nng, 존재__nng, 발생__nng, 정체__nng, 위험__nng, 세계__nng, 

가족:아빠__nng, 엄마__nng, 학교__nng, 가족__nng, 딸__nng, 아들__nng, 아버지__nng, 마을__nng, 집__nng, 아이__nng, 살__vv, 찾__vv, 하__vv, 있__vv, 친구__nng, 영화제__nng, 되__vv, 날__nng, 국제__nng, 사랑__nng, 자신__nng, 시작__nng, 받__vv, 사람__nng, 

공연:비올레타__nnp, 레오노라__nnp, 실황__nng, 토스카__nng, 스카르피__nnp, 알프레도__nnp, 카바__nng, 백작__nng, 로지__nng, 오페라__nng, 공작__nng, 콘서트__nng, 왕자__nng, 여왕__nng, 노__nnp, 공연__nng, 공주__nng, 극장__nng, 무대__nng, 비극__nng, 파리__nng, 왕__nng, 코__nng, 시__nng, 부인__nng, 연인__nng, 여인__nng, 음악__nng, 돈__nng, 도시__nng, 

```

Everything is okay.
That is acceptable answers.

#### 3rd object

```python
import numpy as np; 
import seaborn as sns; 
import matplotlib.pyplot as plt
np.random.seed(0)

# 한글 지원 폰트
sns.set(font='NanumGothic')

# 마이너스 부호 

plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 15))
ax = sns.heatmap(matrix, xticklabels=genre_name, yticklabels=genre_name, annot=True,  cmap='RdYlGn_r')
ax

```

Heatmap is nice.
It was worked.


### Section 2

Yes, there is enuogh comments, i can understand what the code do.
```python
#tfidf를 문서 두개에 적용하게 되면 빈도수 상위 기준 대부분이 df가 같다. 즉 idf가 같다.
#그러니 wordcount로 바꿔 비율을 구해보기로 했다.
word_counts_art = Counter(art)
word_counts_gen = Counter(gen)

#art에 등장한 단어 카운트를 gen에 등장한 카운트로 나눈다
word_ratios1 = {word: count / word_counts_gen[word] if word in word_counts_gen else 1 for word, count in word_counts_art.items() if count > 30}
sorted_word_ratios_art = sorted(word_ratios1.items(), key=lambda x: x[1], reverse=True)
#gen에 등장한 단어 카운트를 art에 등장한 카운트로 나눈다
word_ratios2 = {word: count / word_counts_art[word] if word in word_counts_art else 1 for word, count in word_counts_gen.items() if count > 30}
sorted_word_ratios_gen = sorted(word_ratios2.items(), key=lambda x: x[1], reverse=True)
print('예술영화를 대표하는 단어들:')
```

### Section 3
```python
os.getenv('HOME')+'/aiffel/weat/'+file_name
```

By use os.getenv he prevented path errors from occurring.


### Section 4

```python
from numpy import dot
from numpy.linalg import norm
def cos_sim(i, j):
    return dot(i, j.T)/(norm(i)*norm(j))

def s(w, A, B):
    c_a = cos_sim(w, A)
    c_b = cos_sim(w, B)
    mean_A = np.mean(c_a, axis=-1)
    mean_B = np.mean(c_b, axis=-1)
    return mean_A - mean_B #, c_a, c_b
def weat_score(X, Y, A, B):
    
    s_X = s(X, A, B)
    s_Y = s(Y, A, B)

    mean_X = np.mean(s_X)
    mean_Y = np.mean(s_Y)
    
    std_dev = np.std(np.concatenate([s_X, s_Y], axis=0))
    
    return  (mean_X-mean_Y)/std_dev
```

He understood the formula for calculating the WEAK score well.

### Section 5

```python
from numpy import dot
from numpy.linalg import norm
def cos_sim(i, j):
    return dot(i, j.T)/(norm(i)*norm(j))

def s(w, A, B):
    c_a = cos_sim(w, A)
    c_b = cos_sim(w, B)
    mean_A = np.mean(c_a, axis=-1)
    mean_B = np.mean(c_b, axis=-1)
    return mean_A - mean_B #, c_a, c_b
def weat_score(X, Y, A, B):
    
    s_X = s(X, A, B)
    s_Y = s(Y, A, B)

    mean_X = np.mean(s_X)
    mean_Y = np.mean(s_Y)
    
    std_dev = np.std(np.concatenate([s_X, s_Y], axis=0))
    
    return  (mean_X-mean_Y)/std_dev
```

```
#장르별 키워드도 잘 뽑힌 것 처럼 보이니 그냥 사용하겠다.
attributes = []
for tt in range(len(temp2)) :
    attr = []
    for i in range(min(30,len(temp2[tt]))) :
        attr.append(temp2[tt][i][0])
    attributes.append(attr)
```

Areas divided by intents in Python are called blocks.
It would be nice to have a line break at the end of a block of code.


----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------

