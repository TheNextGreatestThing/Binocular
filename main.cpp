//
//  main.cpp
//  Binocular
//
//  Created by Ethan Henry on 4/13/24.
//

#include <iostream>
#include <limits.h>
#include <unistd.h>
#include "Tools.h"

using namespace cv;
using namespace std;
using namespace AlphaCore;
int totalpairs = 80000;
int totalcycles = 4;

shared_ptr<AlphaCore::CPUFeedForwardNeuralNetwork> Sigmoid;
//string networkLocationOld = "/Users/ethanh/Library/Mobile\ Documents/com\~apple\~CloudDocs/Coding/Binocular/Resources/Sigmoid1.5";
string networkLocationNew = "Resources/Sigmoid1.7";

float frameRate = 1000.0/60.0; // (60 frames per second)
float frame = 0;

bool diagRan = false;

float nAvg = 0;
float aAvg = 0;
float totalNums = 0;

string result = "";
Scalar resColor;

int width = 1280;
int height = 780;

bool mouseClick = false;
float mouseX;
float mouseY;

bool uploadingAnImage = false;

bool naturalTermination = false;

Mat grayPic;

string exePath;

bool checkMouse(int tx, int ty, int bx, int by) {
    return (mouseX <= bx && mouseX >= tx && mouseY <= by && mouseY >= ty);
}


void terminator(int signal) {
    if (!naturalTermination) {
        
        //Sigmoid->save(networkLocationNew);
    }
}

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    //cout << x << " ' " << y << endl;
    mouseX = x;
    mouseY = y;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouseClick = true;
    }
}

int main(int argc, const char * argv[]) {
    char resolved_path[PATH_MAX];

    // Resolve the path of the executable
    if (realpath(argv[0], resolved_path)) {
        string temp = resolved_path;
        for (int i = 0; i < temp.length() - 9; i++) {
            exePath.push_back(temp[i]);
        }
    } else {
        perror("Error resolving executable path");
        return -1;
    }
    
    
    // insert code here...
    Sigmoid.reset(new AlphaCore::CPUFeedForwardNeuralNetwork({6400, 2032, 1012, 1012, 2032, 1}, networkLocationNew));
    Sigmoid->setActivations(0, AlphaCore::sigmoid);
    Sigmoid->setDerivatives(0, AlphaCore::sigmoidder);
    Sigmoid->setActivations(1, AlphaCore::sigmoid);
    Sigmoid->setDerivatives(1, AlphaCore::sigmoidder);
    Sigmoid->setActivations(2, AlphaCore::sigmoid);
    Sigmoid->setDerivatives(2, AlphaCore::sigmoidder);
    Sigmoid->setActivations(3, AlphaCore::sigmoid);
    Sigmoid->setDerivatives(3, AlphaCore::sigmoidder);
    Sigmoid->setActivations(4, AlphaCore::sigmoid);
    Sigmoid->setDerivatives(4, AlphaCore::sigmoidder);
    Sigmoid->setLearningRate(0.00005f);
    Sigmoid->addElementwiseSkipConnection(1, 4);

    
    //Training network
    /*
    signal(SIGINT, terminator);
    signal(SIGTERM, terminator);
    //Set up neural networks
    cout << "Setting Up Network..." << endl;
    
//    Relu->setDerivatives(4, AlphaCore::sigmoidder);

    

//    Sigmoid.reset(new AlphaCore::CPUFeedForwardNeuralNetwork({6400, 2032, 1012, 1012, 2032, 1}, AlphaCore::HE_INIT));
////    Relu->setDerivatives(4, AlphaCore::sigmoidder);
//
//    Sigmoid->setActivations(0, AlphaCore::relu);
//    Sigmoid->setDerivatives(0, AlphaCore::reluder);
//    Sigmoid->setActivations(1, AlphaCore::relu);
//    Sigmoid->setDerivatives(1, AlphaCore::reluder);
//    Sigmoid->setActivations(2, AlphaCore::relu);
//    Sigmoid->setDerivatives(2, AlphaCore::reluder);
//    Sigmoid->setActivations(3, AlphaCore::relu);
//    Sigmoid->setDerivatives(3, AlphaCore::reluder);
//    Sigmoid->setActivations(4, AlphaCore::sigmoid);
//    Sigmoid->setDerivatives(4, AlphaCore::sigmoidder);
    
    
    Sigmoid->setLearningRate(0.00005f);
    Sigmoid->addElementwiseSkipConnection(1, 4);
    Sigmoid->doNanCheck(false);
    
    cout << "Beginning Training..." << endl;
    // Load the image from file
    for (int c = 0; c < totalcycles; c++) {
    for (int pair = 0; pair <= totalpairs; pair++) {
        int randomN = AlphaCore::random(0, 87501);
        int randomA = AlphaCore::random(0, 59152);
        string filePathN = "/Users/ethanh/Downloads/GasHisSDB/80/Normal/Normal-";
        filePathN += to_string(randomN);
        filePathN += ".png";
        string filePathA = "/Users/ethanh/Downloads/GasHisSDB/80/Abnormal/Abnormal-";
        filePathA += to_string(randomA);
        filePathA += ".png";
        Mat normal = imread(filePathN, IMREAD_GRAYSCALE);
        Mat abnormal = imread(filePathA, IMREAD_GRAYSCALE);
        
        // Check if the image was loaded successfully
        while (normal.empty()) {
            int trandomN = AlphaCore::random(0, 59152);
            string filePathN = "/Users/ethanh/Downloads/GasHisSDB/80/Normal/Normal-";
            filePathN += to_string(trandomN);
            filePathN += ".png";
            normal = imread(filePathN, IMREAD_GRAYSCALE);
        }
        while (abnormal.empty()) {
            int trandomA = AlphaCore::random(0, 59152);
            string filePathA = "/Users/ethanh/Downloads/GasHisSDB/80/Abnormal/Abnormal-";
            filePathA += to_string(trandomA);
            filePathA += ".png";
            abnormal = imread(filePathA, IMREAD_GRAYSCALE);
        }
        
        Mat normalizedImageN;
        normal.convertTo(normalizedImageN, CV_32FC1, 1.0 / 255.0);
        Mat flatImageN = normalizedImageN.reshape(1, 1);
        vector<float> normalVec;
        flatImageN.copyTo(normalVec);
        
        Mat normalizedImageA;
        abnormal.convertTo(normalizedImageA, CV_32FC1, 1.0 / 255.0);
        Mat flatImageA = normalizedImageA.reshape(1, 1);
        vector<float> abnormalVec;
        flatImageA.copyTo(abnormalVec);
        
        if (pair % 50 == 0) {
            totalNums++;
            cout << "Cycle " << c + 1 << " of " << totalcycles << endl;
            auto result = Sigmoid->feedforward(move(normalVec));
            nAvg += result[0];
            cout << "Pair: " << pair << endl << result[0] << endl;
            cout << "nAvg: " << nAvg/totalNums << endl;
            result = Sigmoid->feedforward(move(abnormalVec));
            aAvg += result[0];
            cout << "Pair: " << pair << endl << result[0] << endl;
            cout << "aAvg: " << aAvg/totalNums << endl << endl;
        }
        
        Sigmoid->backpropagate(move(normalVec), {1});
        Sigmoid->backpropagate(move(abnormalVec), {0});
        
        // Display the grayscale image
//        imshow("Grayscale Image", normal);
//        waitKey(0);
    }
    }
    //cout << "Saving Network..." << endl;
    naturalTermination = true;
    //Sigmoid->save(networkLocationNew);
     */
    
    string enteredFilepath = "";
    
    float titlescreenduration = 8; //8 seconds long
    string r1 = exePath;
    r1 += "Resources/1.png";
    string r3 = exePath;
    r3 += "Resources/3.png";
    string r4 = exePath;
    r4 += "Resources/4.png";
    string rtc = exePath;
    rtc += "Resources/tc.png";
    string rctri = exePath;
    rctri += "Resources/CTRI.png";
    string rus = exePath;
    rus += "Resources/uploadScreen.png";
    cv::Mat ts = cv::imread(r1);
    Mat noUpload = imread(r3);
    Mat uploadPresent = imread(r4);
    Mat termsandcond = imread(rtc);
    Mat ctri = imread(rctri);
    Mat uploadScreen = imread(rus);

    float cursorA = 0;
        if (ts.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            return -1;
        }
    // Create a named window with the specified size
    cv::namedWindow("Binocular", cv::WINDOW_NORMAL);
    setMouseCallback("Binocular", onMouse);
    Mat uploadedImage;
    // Keep callback state alive throughout the event loop (OpenCV stores this pointer).
    tuple<string&, Mat&, float&> searchCallbackData(enteredFilepath, uploadedImage, frame);
    while (true) {
        Mat typedScreen = uploadScreen.clone();
        int key = waitKey(frameRate);

        if (frame <= titlescreenduration*60) {//Titlescreen for 8 seconds
            // Display title screen image
            cv::imshow("Binocular", ts);
            frame++;
        } else if (frame == titlescreenduration*60 + 1){
            cv::imshow("Binocular", ts);
            namedWindow("Terms And Conditions");
            cv::imshow("Terms And Conditions", termsandcond);
            setMouseCallback("Terms And Conditions", [](int event, int x, int y, int flags, void* userdata){
                    if (event == cv::EVENT_LBUTTONDOWN)
                    {
                        // Left mouse button down event
                        if (x<=295 && x>=179 && y >= 419 && y <= 451) {
                            //Accept button clicked
                            destroyWindow("Terms And Conditions");
                            frame++;
                        } else if (x<=147 && x>=32 && y >= 419 && y <= 451) {
                            //Decline button clicked;
                            destroyAllWindows();
                            terminate();
                        }
                    }
                    return;
                });
        } else if (frame > titlescreenduration*60 + 1 && frame < 679) {
            //No upload yet
            cv::imshow("Binocular", noUpload);
            if (mouseClick == true && mouseX < 855 && mouseX >590 && mouseY < 743 && mouseY > 709) {
                //Selected Choosing The Right Image
                namedWindow("Choosing The Right Image");
                cv::imshow("Choosing The Right Image", ctri);
                setMouseCallback("Choosing The Right Image", [](int event, int x, int y, int flags, void* userdata){
                        if (event == cv::EVENT_LBUTTONDOWN)
                        {
                            // Left mouse button down event
                            if (x<=295 && x>=179 && y >= 419 && y <= 451) {
                                //Ok button clicked
                                destroyWindow("Choosing The Right Image");
                            }
                        }
                        return;
                    });
            } else if (mouseClick == true && checkMouse(90, 240, 467, 690)) {
                //Uploading an image
                namedWindow("Search");
                uploadingAnImage = true;
                setMouseCallback("Search", [](int event, int x, int y, int flags, void* userdata){
                        if (event == cv::EVENT_LBUTTONDOWN)
                        {
                            auto& tp = *static_cast<tuple<string&, Mat&, float&>*>(userdata);
                            string& efp = get<0>(tp);
                            Mat& upI = get<1>(tp);
                            float& fr = get<2>(tp);
                            // Left mouse button down event
                            mouseX = x;
                            mouseY = y;
                            if (checkMouse(207, 339, 353, 377)) {
                                //Cancel button clicked
                                efp = "";
                                uploadingAnImage = false;
                                destroyWindow("Search");
                            } else if (checkMouse(680, 339, 826, 377)) {
                                //Upload button clicked
                                upI = imread(efp);
                                if (upI.empty()) {
                                    //
                                } else {
                                    diagRan = false;
                                    destroyWindow("Search");
                                    uploadingAnImage = false;
                                    fr = 680;
                                }
                            }
                        }
                        return;
                    }, &searchCallbackData);
            }
            
            if (uploadingAnImage) {
                Size textSize = getTextSize(enteredFilepath, FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
                if ((key == 127 || key == 8) && enteredFilepath.length() > 0) {
                    enteredFilepath.pop_back();
                } else if (key != -1 && textSize.width < 780 && key != 10 && key != 13) {
                    enteredFilepath.push_back(static_cast<char>(key));
                }
                putText(typedScreen, enteredFilepath, Point(100, 280), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
                
                int cursorX = 100 + textSize.width;
                int cursorY = 280;
                float appearC = abs(255*sin(cursorA)) > 100;
                if (appearC) {
                    line(typedScreen, Point(cursorX, cursorY - textSize.height), Point(cursorX, cursorY), Scalar(255, 255, 255));
                }
                imshow("Search", typedScreen);
                cursorA+=0.05;
                if (cursorA >= 2*3.12149*600) {
                    cursorA = 0;
                }
            }
            
            frame++;
            cap(frame, titlescreenduration*60 + 100);
        } else if (frame > 679) {
           // cout << mouseX << endl << mouseY << endl << endl;
            //Uploaded
            while (frame <= 700) {
                int tkey = waitKey(frameRate);
                Mat currentScreen = uploadPresent.clone();
                Mat oldPicture = imread(enteredFilepath);
                Mat newPicture;
                resize(oldPicture, newPicture, Size(mapf(frame, 680, 700, 5, 350), mapf(frame, 680, 700, 5, 350)));
                placeImage(currentScreen, newPicture, 274, 470);
            
                imshow("Binocular", currentScreen);
                frame+=1.5;
            }
            Mat currentScreen = uploadPresent.clone();
            Mat picture = imread(enteredFilepath);
            resize(picture, picture, Size(350, 350));
            placeImage(currentScreen, picture, 274, 470);
            
            if (mouseClick && checkMouse(100, 234, 140, 269)) {
                frame = titlescreenduration*60 + 10;
            }

            if (mouseClick && checkMouse(182, 111, 384, 176)) {
                //Run Diagnostic button click
                diagRan = true;
                Mat realPic = imread(enteredFilepath);
                grayPic = convertToGrayscale(realPic);
                float output = Sigmoid->feedforward(convertTo1DVector(grayPic))[0];
                //cout << output << endl;
                if (output < 0.5) {
                    result = "Abnormal";
                    resColor = Scalar(0,40, 255);//BGR format
                } else if (output > 0.5){
                    result = "Normal";
                    resColor = Scalar(20,255, 40);//BGR format
                } else if (output == 0.5) {
                    result = "Inconclusive";
                    resColor = Scalar(100,100, 100);//BGR format
                }
            }
        
            if (diagRan) {
                cv::Point location(860, 170);
                int fontFace = cv::FONT_HERSHEY_TRIPLEX;
                double fontScale = 0.85;
                int thickness = 1;
                addTextToImage(currentScreen, result, location, fontFace, fontScale, resColor, thickness);
                
                Mat normal = imread("Resources/Normal-42760.png");
                Mat grayNormal = convertToGrayscale(normal);
                float control = Sigmoid->feedforward(convertTo1DVector(grayNormal))[0];
                
                for (int quadrant = 1 ; quadrant <= 4; quadrant++) {
                    // Start each comparison from the same reference image to avoid
                    // cumulative replacement across quadrants.
                    Mat quadrantImage = grayNormal.clone();
                    replaceQuadrant(quadrantImage, grayPic, quadrant);
                
                    string text = "Q";
                    text += to_string(quadrant);
                    text += ": ";
                    float divergence = control - Sigmoid->feedforward(convertTo1DVector(quadrantImage))[0];
                    text += to_string(divergence);
                    Scalar color;
                    if (divergence < 0.2) {
                        color = Scalar(20, 255, 40);
                    } else if (divergence >= 0.3 ) {
                        color = Scalar(0, 40, 255);
                    } else {
                        color = Scalar(255, 255, 255);
                    }
                    addTextToImage(currentScreen, text, Point(845, 210 + quadrant * 55), fontFace, fontScale,   color, thickness);
                }
                
                drawCross(currentScreen, Point(274, 470), 400, Scalar(0, 0, 0), 4);
                drawCross(currentScreen, Point(274, 470), 400, Scalar(255, 255, 255));
            }
            //274, 270
            //Change upload screen (eg. to not include Choosing the right image and also make it a lil cleaner)
            //Get rid of try again and add an actual error screen with restrictions like the 80 x 80 one
            
            imshow("Binocular", currentScreen);
            frame++;
            cap(frame, 1200);
        }
        resizeWindow("Binocular", width, height);
        
        if (key == 27) {
            //When escape key pressed
            break;
        }
        mouseClick = false;

    }
        cv::destroyAllWindows();
    
    return 0;
}
