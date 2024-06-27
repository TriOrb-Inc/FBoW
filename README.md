# Fast Bag of Words

**This is a modified version of [the original Fast Bag of Words](https://github.com/rmsalinas/fbow) by [@rmsalinas](https://github.com/rmsalinas).**

FBoW (Fast Bag of Words) is an extremely optimized version of the [DBoW2](https://github.com/dorian3d/DBoW2)/[DBoW3](https://github.com/rmsalinas/DBow3) libraries.  
The library is highly optimized to speed up the Bag of Words creation using AVX,SSE and MMX instructions.  
In loading a vocabulary, FBoW is about 80x faster than DBoW2.  
In transforming an image into Bag of Words using on machines with AVX instructions, it is about 6.4x faster.  

## Build

```bash
$ git clone https://github.com/shinsumicco/FBoW.git
$ cd FBoW && mkdir build && cd build
$ cmake .. -DBUILD_TESTS=ON -DBUILD_UTILS=ON
$ make
```

## License

This software is distributed under MIT License.  
See the [LICENSE](./LICENSE).

　　
## 追記
### 変更点
cuda-efficient-featuresを使用してBADとHASH_SIFT特徴量を作成  
### 実行手順
①使用するファイルを一つのディレクトリに保存  
②cd FBow/build/utils  
③./fbow_dump_features bad256 save_file /picture_dir  
&emsp;※./fbow_dump_features 使用特徴量 保存ファイル名 使用する画像のディレクトリ  
④./fbow_create_vocabulary save_file vocablary.fbow  
&emsp;※./fbow_create_vocabulary ③で作成したファイル 生成する辞書ファイル名  
  
### 未実装
複数のディレクトリからファイルを取得