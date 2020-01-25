# HanBert-nsmc

Naver movie review sentiment classification with HanBert

## Tokenization

- tokenization_hanbert.py의 `HanBertTokenizer`를 사용하면 됩니다.

### How to use

- [HanBert 다운로드 페이지](https://twoblockai.com/2020/01/22/hanbert%eb%a5%bc-%ea%b3%b5%ea%b0%9c%ed%95%a9%eb%8b%88%eb%8b%a4/)로 가서 다운로드
- tokenizer 관련 c++ 파일 로컬에 옮기기

```bash
tar xvfz hanbert.tar.gz
cd HanBert-54kN/
sudo mkdir /usr/local/moran
cp usr_local_moran/* /usr/local/moran
```

- HanBert-torch라는 폴더 만든 후, 그 안에 vocab_54k.txt 넣기

- 해당 파이썬 실행

```python
>>> from tokenization_hanbert import HanBertTokenizer
>>> tokenizer = HanBertTokenizer.from_pretrained('HanBert-torch', do_lower_case=False)
>>> text = "나는 걸어가고 있는 중입니다. 나는걸어 가고있는 중입니다. 잘 분류되기도 한다. 잘 먹기도 한다."
>>> tokenizer.tokenize(text)
>>> tokenizer.encode_plus(text)
```
