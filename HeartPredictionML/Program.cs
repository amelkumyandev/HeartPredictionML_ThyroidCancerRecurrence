using Microsoft.ML;
using Microsoft.ML.Data;

namespace HeartDiseaseClassificationComparison
{
    // Input data schema
    public class HeartData
    {
        [LoadColumn(0)] public float Age { get; set; }
        [LoadColumn(1)] public float Gender { get; set; }
        [LoadColumn(2)] public float BloodPressure { get; set; }
        [LoadColumn(3)] public float Cholesterol { get; set; }
        [LoadColumn(4)] public float HeartRate { get; set; }
        [LoadColumn(5)] public float QuantumPatternFeature { get; set; }
        [LoadColumn(6)] public float HeartDisease { get; set; }  // 0 or 1
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create ML context
            var mlContext = new MLContext(seed: 0);

            // File paths for training and test data (adjust as needed)
            string trainDataPath = "HeartPredictionQuantumDataset_Train.csv";
            string testDataPath = "HeartPredictionQuantumDataset_Test.csv";

            // Load data
            IDataView trainData = mlContext.Data.LoadFromTextFile<HeartData>(
                trainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testData = mlContext.Data.LoadFromTextFile<HeartData>(
                testDataPath, hasHeader: true, separatorChar: ',');

            // Define feature columns (all except the label)
            string[] featureCols = new[]
            {
                nameof(HeartData.Age), nameof(HeartData.Gender), nameof(HeartData.BloodPressure),
                nameof(HeartData.Cholesterol), nameof(HeartData.HeartRate), nameof(HeartData.QuantumPatternFeature)
            };

            // Data processing pipeline: convert label to bool, concatenate features, normalize features
            var dataProcessPipeline = mlContext.Transforms.Conversion
                                        .ConvertType(outputColumnName: "Label", inputColumnName: "HeartDisease", outputKind: DataKind.Boolean)
                                    .Append(mlContext.Transforms.Concatenate("Features", featureCols))
                                    .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Fit the data processing pipeline on training data
            var dataPrepTransformer = dataProcessPipeline.Fit(trainData);
            IDataView transformedTrain = dataPrepTransformer.Transform(trainData);
            IDataView transformedTest = dataPrepTransformer.Transform(testData);


            // List to hold evaluation metrics for each model
            var results = new List<(string ModelName, BinaryClassificationMetrics Metrics, bool Probabilistic)>();

            // 1. SDCA Logistic Regression
            var sdcaTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                                    labelColumnName: "Label", featureColumnName: "Features");
            var sdcaModel = sdcaTrainer.Fit(transformedTrain);
            var sdcaPredictions = sdcaModel.Transform(transformedTest);
            BinaryClassificationMetrics sdcaMetrics = mlContext.BinaryClassification.Evaluate(
                                                        sdcaPredictions, labelColumnName: "Label", scoreColumnName: "Score", predictedLabelColumnName: "PredictedLabel");
            results.Add(("SDCA Logistic Regression", sdcaMetrics, true));

            // 2. L-BFGS Logistic Regression
            var lbfgsTrainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                                     labelColumnName: "Label", featureColumnName: "Features");
            var lbfgsModel = lbfgsTrainer.Fit(transformedTrain);
            var lbfgsPredictions = lbfgsModel.Transform(transformedTest);
            BinaryClassificationMetrics lbfgsMetrics = mlContext.BinaryClassification.Evaluate(
                                                        lbfgsPredictions, labelColumnName: "Label");
            results.Add(("L-BFGS Logistic Regression", lbfgsMetrics, true));

            // 3. Averaged Perceptron
            var apTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                                    labelColumnName: "Label", featureColumnName: "Features");
            var apModel = apTrainer.Fit(transformedTrain);
            var apPredictions = apModel.Transform(transformedTest);
            // Perceptron produces uncalibrated scores, use EvaluateNonCalibrated
            var apMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
                                apPredictions, labelColumnName: "Label", scoreColumnName: "Score", predictedLabelColumnName: "PredictedLabel");
            results.Add(("Averaged Perceptron", apMetrics, false));

            // 4. Linear SVM
            var svmTrainer = mlContext.BinaryClassification.Trainers.LinearSvm(
                                    labelColumnName: "Label", featureColumnName: "Features");
            var svmModel = svmTrainer.Fit(transformedTrain);
            var svmPredictions = svmModel.Transform(transformedTest);
            var svmMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
                                svmPredictions, labelColumnName: "Label");
            results.Add(("Linear SVM", svmMetrics, false));

            // 5. Local Deep SVM (non-linear SVM)
            var ldSvmTrainer = mlContext.BinaryClassification.Trainers.LdSvm(
                                    labelColumnName: "Label", featureColumnName: "Features");
            var ldSvmModel = ldSvmTrainer.Fit(transformedTrain);
            var ldSvmPredictions = ldSvmModel.Transform(transformedTest);
            var ldSvmMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
                                    ldSvmPredictions, labelColumnName: "Label");
            results.Add(("Local Deep SVM", ldSvmMetrics, false));

            // 6. FastTree (Gradient Boosted Trees)
            // (FastTree is in Microsoft.ML.FastTree NuGet)
            var fastTreeTrainer = mlContext.BinaryClassification.Trainers.FastTree(
                                        labelColumnName: "Label", featureColumnName: "Features");
            var fastTreeModel = fastTreeTrainer.Fit(transformedTrain);
            var fastTreePredictions = fastTreeModel.Transform(transformedTest);
            var fastTreeMetrics = mlContext.BinaryClassification.Evaluate(
                                        fastTreePredictions, labelColumnName: "Label");
            results.Add(("FastTree GBDT", fastTreeMetrics, true));

  
            var lightGbmTrainer = mlContext.BinaryClassification.Trainers.LightGbm(
                                        labelColumnName: "Label", featureColumnName: "Features");
            var lightGbmModel = lightGbmTrainer.Fit(transformedTrain);
            var lightGbmPredictions = lightGbmModel.Transform(transformedTest);
            var lightGbmMetrics = mlContext.BinaryClassification.Evaluate(
                                        lightGbmPredictions, labelColumnName: "Label");
            results.Add(("LightGBM GBDT", lightGbmMetrics, true));

            // 9. GAM (Generalized Additive Model)
            var gamTrainer = mlContext.BinaryClassification.Trainers.Gam(
                                    labelColumnName: "Label", featureColumnName: "Features");
            var gamModel = gamTrainer.Fit(transformedTrain);
            var gamPredictions = gamModel.Transform(transformedTest);
            var gamMetrics = mlContext.BinaryClassification.Evaluate(
                                    gamPredictions, labelColumnName: "Label");
            results.Add(("GAM (Generalized Additive)", gamMetrics, true));

            // Print header
            Console.WriteLine("\nModel Performance on Test Data:");
            Console.WriteLine("{0,-30} {1,8} {2,6} {3,6} {4,9} {5,7}   ConfusionMatrix (TP/FP/FN/TN)",
                              "Model", "Accuracy", "AUC", "F1", "Precision", "Recall");
            foreach (var (ModelName, Metrics, Probabilistic) in results)
            {
                // Use "N/A" for AUC if model did not produce probabilities
                string aucStr = Probabilistic ? Metrics.AreaUnderRocCurve.ToString("F3") : "N/A";
                // We take Precision/Recall for the positive class (Label=true)
                string precStr = Metrics.PositivePrecision.ToString("F3");
                string recStr = Metrics.PositiveRecall.ToString("F3");
                // Confusion matrix values
                var cm = Metrics.ConfusionMatrix.Counts; // 2x2 matrix [actual][predicted]
                long tn = (long)cm[0][0];
                long fp = (long)cm[0][1];
                long fn = (long)cm[1][0];
                long tp = (long)cm[1][1];
                Console.WriteLine("{0,-30} {1,8:F3} {2,6} {3,6:F3} {4,9:F3} {5,7:F3}   TP={6}, FP={7}, FN={8}, TN={9}",
                                  ModelName, Metrics.Accuracy, aucStr, Metrics.F1Score,
                                  Metrics.PositivePrecision, Metrics.PositiveRecall,
                                  tp, fp, fn, tn);
            }
        }
    }
}
