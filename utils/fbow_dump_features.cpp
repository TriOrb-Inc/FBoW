/**

The MIT License

Copyright (c) 2017 Rafael Muñoz-Salinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "cmd_line_parser.h"
#include "dir_reader.h"

#include <iostream> //汎用入出力
#include <fstream> //ファイル入出力
#include <vector> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/imgproc.hpp> //追記
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

//追記
#include <cuda_efficient_features/cuda_efficient_descriptors.h>
#include <cuda_efficient_features/cuda_efficient_descriptors.h>
#include <cuda_efficient_features/cuda_efficient_features.h>

std::vector<cv::Mat> loadFeatures(const std::vector<std::string>& path_to_images, const std::string& descriptor = "orb") {
    cv::Ptr<cv::Feature2D> feat_detector = nullptr; //使用する特徴の変数

    /*追記　変数の用意*/
    cv::Ptr<cv::cuda::EfficientFeatures> efficient_detector = nullptr;

    // どの特徴量を使うか
    if (descriptor == "orb") feat_detector = cv::ORB::create(2000);
    else if (descriptor == "brisk") feat_detector = cv::BRISK::create();
    else if (descriptor == "akaze") feat_detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);

    else if (descriptor == "bad256") efficient_detector = cv::cuda::EfficientFeatures::create(100000,1.2f,8,0,20,15,cv::cuda::EfficientFeatures::DescriptorType::BAD_256);//追記 bad256用
    else if (descriptor == "bad512") efficient_detector = cv::cuda::EfficientFeatures::create(100000,1.2f,8,0,20,15,cv::cuda::EfficientFeatures::DescriptorType::BAD_512);//追記 bad256用
    else if (descriptor == "hash_sift256") efficient_detector = cv::cuda::EfficientFeatures::create(100000,1.2f,8,0,20,15,cv::cuda::EfficientFeatures::DescriptorType::HASH_SIFT_256);//追記 hash_sift256用
    else if (descriptor == "hash_sift512") efficient_detector = cv::cuda::EfficientFeatures::create(100000,1.2f,8,0,20,15,cv::cuda::EfficientFeatures::DescriptorType::HASH_SIFT_512);//追記 hash_sift256用

#ifdef USE_CONTRIB
    else if (descriptor == "surf") feat_detector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif
    else throw std::runtime_error("invalid descriptor: " + descriptor);

    assert(!descriptor.empty()); //条件を満たさないときはプログラムを異常終了
    std::vector<cv::Mat> features; //特徴量用の変数

    std::cout << "extracting features ..." << std::endl;
    for (const auto& path_to_image : path_to_images) { //特徴抽出
        std::vector<cv::KeyPoint> keypoints; //特徴点用の変数
        cv::Mat descriptors; // 特徴量用の変数

        std::cout << "reading image: " << path_to_image << std::endl;
        cv::Mat image = cv::imread(path_to_image, 0);//データをグレースケールで読み込み
        if (image.empty()) { //データが何もなかったら
            std::cerr << "could not open image: " << path_to_image << std::endl;
            continue;
        }
        // feat_detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors); // 指定した特徴を用いて特徴点→特徴量を算出
        // 引数 image:特徴点を検出したい画像　cv::Mat():マスク画像(今回は使用しないため入力画像がそのまま検出に使われる) keypoints:検出された特徴点用の出力変数 descriptors:特徴点に対応する特徴量用の出力変数
        
        /*追記 どちらの変数に値を入れるか選択*/
        if (!feat_detector){ //feat_detectorがnullptrのとき？
            std::cout << "used efficient_detector" << std::endl; //if文確認用(後でコメント化)
            efficient_detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        }
        else{
            std::cout << "used feat_detector" << std::endl; //else文確認用(後でコメント化) 
            feat_detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        }

        std::cout << "extracted features: total = " << keypoints.size() << std::endl; //検出した特徴量の数を表示
        
        if (keypoints.size()<100){//追記 特徴量が100よりも少ない場合読み飛ばす
            continue;
        }

        features.push_back(descriptors); // 特徴量の内容をfeaturesに追記
        std::cout << "done detecting features" << std::endl;
    }
    return features;
}

void saveToFile(const std::string& filename, const std::vector<cv::Mat>& features, std::string desc_name, bool rewrite = true) {
    if (!rewrite) { //新しくファイルを作成する場合？
        std::fstream ifile(filename, std::ios::binary);
        if (ifile.is_open()) {
            std::runtime_error("output file " + filename + " already exists");
        }
    }
    std::ofstream ofile(filename, std::ios::binary); //filenameで指定したファイルをbinary形式で開く
    if (!ofile.is_open()) { //ファイルが開けなかった場合
        std::cerr << "could not create an output file: " << filename << std::endl;
        exit(1);
    }

    char _desc_name[20];
    desc_name.resize(std::min(size_t(19), desc_name.size())); //特徴量名の長さを変更(最大19文字)?
    strcpy(_desc_name, desc_name.c_str()); //_desc_nameにdesc_nameの内容をchar*型でコピー
    ofile.write(_desc_name, 20); //_desc_nameの内容を20文字以内でofileに書き込み

    uint32_t size = features.size(); //特徴量の大きさ
    ofile.write((char*) &size, sizeof(size)); // 特徴量の大きさをファイルに書き込み
    for (const auto& f : features) { //範囲for文 featuresの内容をひとつづつ取り出して使用
        if (!f.isContinuous()) { //特徴量が連続でない場合プログラムを終了
            std::cerr << "matrices should be continuous" << std::endl;
            exit(0); 
        }
        uint32_t aux = f.cols; //特徴量行列の列数
        ofile.write((char*)&aux, sizeof(aux)); //列要素の書き込み
        aux = f.rows; ////特徴量行列の行数
        ofile.write((char*)&aux, sizeof(aux)); //行要素の書き込み
        aux = f.type(); //特徴量行列の型?
        ofile.write((char*)&aux, sizeof(aux)); //型の書き込み?
        ofile.write((char*)f.ptr<uchar>(0), f.total() * f.elemSize()); //特徴量の書き込み？
    }
}

int main(int argc, char** argv) {
    try {
        CmdLineParser cml(argc, argv); //cmlというオブジェクトを作成
        if (cml["-h"] || argc < 4) { // エラーが発生した場合
            std::cerr << "Usage: DESCRIPTOR_NAME (= orb, brisk, akaze, surf(contrib)) FEATURE_OUTPUT IMAGES_DIR" << std::endl;
            std::cerr << std::endl;
            std::cerr << "First step of creating a vocabulary is extracting features from a set of images." << std::endl;
            std::cerr << "We save them to a file for next step." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE; //異常終了を返す
        }

        //定数の定義
        const std::string descriptor = argv[1]; //引数一つ目:どの特徴量を使うか
        const std::string output = argv[2]; //引数二つ目:保存ファイル名

        auto images = DirReader::read(argv[3]); // 引数三つ目:画像データの読み込み(dir_reader.hのread関数)
        std::vector<cv::Mat> features = loadFeatures(images, descriptor); //特徴量を検出

        std::cout << "saving the features: " << argv[2] << std::endl;

        saveToFile(argv[2], features, descriptor); //ファイルへの書き込み
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl; //エラーメッセージを表示
    }

    // std::cout << "successfully" << std::endl; // 追記 プログラムがうまくいったかの確認
    return EXIT_SUCCESS; //正常終了を返す
}