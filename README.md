# HanBert-nsmc

- HanBert를 이용한 네이버 영화 리뷰 감정 분석 (sentiment classification)
- 🤗Huggingface Tranformers🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.4.0
- transformers==2.7.0

## Details

기본적인 사용법은 [HanBert-Transformers](https://github.com/monologg/HanBert-Transformers)를 참고

### Prerequisite

- Tokenizer의 경우 현재 Ubuntu에서만 사용 가능
- HanBert Model 다운로드 (Pretrained weight + Tokenizer) 및 압축 해제
  - [HanBert-54kN-torch](https://drive.google.com/open?id=1LUyrnhuNC3e8oD2QMJv8tIDrXrxzmdu4)
  - [HanBert-54kN-IP-torch](https://drive.google.com/open?id=1wjROsuDKoJQx4Pu0nqSefVDs3echKSXP)

### Usage

```bash
$ python3 main.py --model_type hanbert \
                  --model_name_or_path HanBert-54kN-torch\
                  --do_train \
                  --do_eval

$ python3 main.py --model_type hanbert \
                  --model_name_or_path HanBert-54kN-IP-torch\
                  --do_train \
                  --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results

Hyperparameter는 main.py에 있는 것을 그대로 사용하였습니다

|                   | Accuracy (%) |
| ----------------- | ------------ |
| HanBert-54kN      | **90.16**    |
| HanBert-54kN-IP   | 88.72        |
| KoBERT            | 89.63        |
| DistilKoBERT      | 88.41        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

## References

- [HanBert](https://github.com/tbai2019/HanBert-54k-N)
- [NSMC result on KoBERT](https://github.com/monologg/KoBERT-nsmc)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC dataset](https://github.com/e9t/nsmc)
