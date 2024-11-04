# IMCEC
Image-Based malware classification using ensemble of CNN architectures (IMCEC)


Project Overview

This project investigates advanced malware classification using deep learning models and SVMs (Support Vector Machines), with particular attention to obfuscation-resistant techniques. The project leverages transfer learning (TL) and fine-tuning (FT) on models like VGG16 and ResNet, and explores ensemble methods to improve detection accuracy. The goal is to provide robust malware detection solutions that remain effective even against common obfuscation methods like packing and salting.

Key Features

	•	Model Variants: Includes multiple configurations of popular architectures (e.g., VGG16 and ResNet) combined with softmax and SVM classifiers.
	•	Obfuscation Resistance: Evaluates model performance against obfuscated malware samples, including packed and salted obfuscation techniques.
	•	ROC Curves for Evaluation: Provides detailed ROC curves to assess the performance of each model configuration, offering insights into model robustness and classification accuracy.
	•	Image Generation for Malware: Implements a method for generating visual representations of malware, facilitating image-based classification.

Project Structure

	•	VGG16-FT-Softmax: Fine-tuned VGG16 model with a softmax classifier for multiclass classification.
	•	ResNet-FT-SVM: Fine-tuned ResNet model combined with an SVM for enhanced classification accuracy.
	•	Obfuscation-Packed and Obfuscation-Salted: Directories containing datasets or configurations related to obfuscation techniques, testing model resilience.
	•	Ensemble: Combines multiple models in an ensemble approach to achieve higher accuracy.
	•	ROC Curves: Precomputed ROC curves for each model configuration, allowing for a comprehensive performance comparison.

Technical Requirements

	•	TensorFlow and Keras: Core libraries for model training and evaluation.
	•	Scikit-learn: Used for integrating SVM classifiers.
	•	Matplotlib and Seaborn: For visualization, including ROC curves and other performance metrics.

How It Works

	1.	Malware Image Generation: The project uses malware binaries to create images, stored and labeled in the dataset. The generated images are then processed by CNN-based models.
	2.	Model Training and Evaluation: Each model architecture (e.g., VGG16, ResNet) is trained with fine-tuning or transfer learning, then evaluated with SVM or softmax classifiers. Obfuscation techniques are applied to test model resilience.
	3.	ROC Curve Analysis: ROC curves are generated for each configuration, providing insights into the true positive rate and false positive rate, which are critical for malware detection accuracy.

Results and Findings

	•	Obfuscation Resilience: Models are tested against obfuscation techniques, with findings showing the effectiveness of fine-tuned CNNs in handling obfuscation such as packing and salting.
	•	Comparative Performance: ROC curves for each model configuration demonstrate the relative effectiveness of each approach, showing that certain architectures combined with SVM classifiers yield higher accuracy.
	•	Ensemble Superiority: The ensemble model outperforms individual models in many cases, suggesting that combining model outputs leads to better overall performance.

Conclusion

This project advances malware classification by employing deep learning models capable of resisting obfuscation techniques. By combining transfer learning, fine-tuning, and SVMs, the approach enhances malware detection accuracy and robustness.

Cite This Work

If you find this project helpful in your research or work, please consider citing our paper:

Danish Vasan, Mamoun Alazab, Sobia Wassan, Babak Safaei, Qin Zheng,
Image-Based malware classification using ensemble of CNN architectures (IMCEC),
Computers & Security,
Volume 92,
2020,
101748,
ISSN 0167-4048,
https://doi.org/10.1016/j.cose.2020.101748.
(https://www.sciencedirect.com/science/article/pii/S016740482030033X)
Abstract: Both researchers and malware authors have demonstrated that malware scanners are unfortunately limited and are easily evaded by simple obfuscation techniques. This paper proposes a novel ensemble convolutional neural networks (CNNs) based architecture for effective detection of both packed and unpacked malware. We have named this method Image-based Malware Classification using Ensemble of CNNs (IMCEC). Our main assumption is that based on their deeper architectures different CNNs provide different semantic representations of the image; therefore, a set of CNN architectures makes it possible to extract features with higher qualities than traditional methods. Experimental results show that IMCEC is particularly suitable for malware detection. It can achieve a high detection accuracy with low false alarm rates using malware raw-input. Result demonstrates more than 99% accuracy for unpacked malware and over 98% accuracy for packed malware. IMCEC is flexible, practical and efficient as it takes only 1.18 s on average to identify a new malware sample.
Keywords: Malware; Cybersecurity; Deep learning; Transfer learning; Fine-tuning; SVMs; Softmax; Ensemble of CNNs
