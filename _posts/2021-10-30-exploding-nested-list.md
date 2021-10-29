---
title: 데이터 프레임 내에 nested된 리스트를 확장하기
tags: [Nested List, Explode, pandas]
category: Pandas
aside:
  toc: true
show_category: true
---

Pandas에서 Nested list를 펼치려면 폭파시켜버리면 됩니다. 💣

<!--more-->

## 🌃 배경

데이터프레임을 다루다보면 컬럼 내에 리스트가 Nested된 형태가 가끔 있습니다. 자주 겪을 수 있는 케이스는 아니라 `pandas` 내에 이걸 처리하는 메서드가 있을거란 생각을 안해봤는데요. 이런 케이스입니다.

```python
>>> df = pd.DataFrame({"col1":["A","B"], "col2":[["a", "b"],["b", "c"]]})

>>> df
  col1    col2
0    A  [a, b]
1    B  [b, c]
```

이 형태로 만들어져 있다면 둘 중 하나일겁니다. 이 형태를 그대로 쓰거나, `col2` 컬럼을 펼쳐서 총 네 개의 행으로 이루어진 데이터프레임을 만들거나.

## 😎 해결

### Numpy만 사용하면

처음엔 `numpy`만을 사용해서 데이터프레임을 재구성했었습니다.

```python
>>> vals = np.array(df["col2"].tolist())
>>> key = np.repeat(df["col1"], vals.shape[1])
>>> pd.DataFrame(np.column_stack((key, vals.ravel())), columns=df.columns)

  col1 col2
0    A    a
1    A    b
2    B    b
3    B    c
```

컨셉은 간단합니다. Nested 되어 있는 컬럼을 `ravel()` 메서드를 통해 펼치고, Key가 되는 컬럼을 Nested 되어 있는 리스트의 길이만큼 반복하여 column-wise하게 붙여주기만 했습니다.

### Pandas를 사용하면

`numpy`만을 사용하던 중 `pandas`에 이런 케이스를 처리해주는 메서드가 있는 것을 알았습니다. 간단하게 한 줄만 쓰면 똑같은 결과를 얻을 수 있습니다.

```python
>>> df.explode("col2", ignore_index=True)

  col1 col2
0    A    a
1    A    b
2    B    b
3    B    c
```

속도 측면에서 비교를 해보지는 않았습니다만, 최근 `pandas 1.3.0` 버전 부터는 여러 칼럼을 explode해주는 기능을 지원한다고 합니다. 자세한 내용은 [Pandas 공식 문서](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html)를 확인해 보시길 바랍니다.