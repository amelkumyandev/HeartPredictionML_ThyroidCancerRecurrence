using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetClassification
{
    class Program
    {
        // Input data class mapping to the CSV columns
        public class ThyroidData
        {
            [LoadColumn(0)] public float Age { get; set; }
            [LoadColumn(1)] public string Gender { get; set; }
            [LoadColumn(2)] public string HxRadiothreapy { get; set; }
            [LoadColumn(3)] public string Adenopathy { get; set; }
            [LoadColumn(4)] public string Pathology { get; set; }
            [LoadColumn(5)] public string Focality { get; set; }
            [LoadColumn(6)] public string Risk { get; set; }
            [LoadColumn(7)] public string T { get; set; }
            [LoadColumn(8)] public string N { get; set; }
            [LoadColumn(9)] public string M { get; set; }
            [LoadColumn(10)] public string Stage { get; set; }
            [LoadColumn(11)] public string Response { get; set; }
            [LoadColumn(12)] public string Recurred { get; set; }  // Label in text form ("Yes"/"No")
        }

        static void Main(string[] args)
        {
            // 1. Create MLContext
            MLContext mlContext = new MLContext(seed: 0);

            // 2. Load data
            string dataPath = "filtered_thyroid_data.csv";
            IDataView dataView = mlContext.Data.LoadFromTextFile<ThyroidData>(
                dataPath, hasHeader: true, separatorChar: ',');

            // 3. Define data preparation pipeline
            // Map "Yes"/"No" in Recurred to a boolean Label (true for "Yes", false for "No")
            var labelMapping = new Dictionary<string, bool> { { "Yes", true }, { "No", false } };
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValue<string, bool>(
                    outputColumnName: "Label", inputColumnName: nameof(ThyroidData.Recurred),
                    keyValuePairs: labelMapping, treatValuesAsKeyType: false)
                // One-hot encode categorical string features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("GenderEncoded", nameof(ThyroidData.Gender)),
                    new InputOutputColumnPair("HxRadioEncoded", nameof(ThyroidData.HxRadiothreapy)),
                    new InputOutputColumnPair("AdenopathyEncoded", nameof(ThyroidData.Adenopathy)),
                    new InputOutputColumnPair("PathologyEncoded", nameof(ThyroidData.Pathology)),
                    new InputOutputColumnPair("FocalityEncoded", nameof(ThyroidData.Focality)),
                    new InputOutputColumnPair("RiskEncoded", nameof(ThyroidData.Risk)),
                    new InputOutputColumnPair("T_Encoded", nameof(ThyroidData.T)),
                    new InputOutputColumnPair("N_Encoded", nameof(ThyroidData.N)),
                    new InputOutputColumnPair("M_Encoded", nameof(ThyroidData.M)),
                    new InputOutputColumnPair("StageEncoded", nameof(ThyroidData.Stage)),
                    new InputOutputColumnPair("ResponseEncoded", nameof(ThyroidData.Response))
                }))
                // Concatenate all feature columns into one Features vector
                .Append(mlContext.Transforms.Concatenate("Features",
                    "Age", "GenderEncoded", "HxRadioEncoded", "AdenopathyEncoded",
                    "PathologyEncoded", "FocalityEncoded", "RiskEncoded",
                    "T_Encoded", "N_Encoded", "M_Encoded", "StageEncoded", "ResponseEncoded"))
                // Cache data in memory to speed up training for multiple models
                .AppendCacheCheckpoint(mlContext);

            // 4. Split data into training and testing sets
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3);
            IDataView trainData = split.TrainSet;
            IDataView testData = split.TestSet;

            // Fit the data prep pipeline and transform the data
            var dataPrepTransformer = dataProcessPipeline.Fit(trainData);
            IDataView transformedTrainData = dataPrepTransformer.Transform(trainData);
            IDataView transformedTestData = dataPrepTransformer.Transform(testData);

            // 5. Define classification trainers to evaluate (name, estimator, isCalibrated)
            var trainers = new List<(string name, IEstimator<ITransformer> estimator, bool isCalibrated)>
            {
                ("Logistic Regression (SDCA)", mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"), true),
                ("Logistic Regression (LBFGS)", mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"), true),
                ("FastTree (Gradient Boosted Trees)", mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"), true),
                ("FastForest (Random Forest)", mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"), false),
                ("Linear SVM", mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features"), false),
                ("Averaged Perceptron", mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features"), false)
            };

            // 6. Train and evaluate each model with exception handling
            foreach (var (name, estimator, isCalibrated) in trainers)
            {
                try
                {
                    Console.WriteLine($"\nTraining model: {name}");
                    // Train model on preprocessed data
                    ITransformer model = estimator.Fit(transformedTrainData);
                    // Use model to make predictions on test data
                    IDataView predictions = model.Transform(transformedTestData);

                    // Evaluate predictions
                    BinaryClassificationMetrics metrics;
                    if (isCalibrated)
                    {
                        // Calibrated model (probability column exists)
                        metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");
                    }
                    else
                    {
                        // Uncalibrated model (no probability column)
                        metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, labelColumnName: "Label");
                    }

                    // Print evaluation metrics
                    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                    Console.WriteLine($"AUC (Area Under ROC Curve): {metrics.AreaUnderRocCurve:P2}");
                    Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

                    // Print confusion matrix (TP, TN, FP, FN)
                    var confusionMatrix = metrics.ConfusionMatrix;
                    if (confusionMatrix?.Counts is IReadOnlyList<IReadOnlyList<double>> cm && cm.Count == 2 && cm[0].Count == 2)
                    {
                        double trueNegatives = cm[0][0];
                        double falsePositives = cm[0][1];
                        double falseNegatives = cm[1][0];
                        double truePositives = cm[1][1];
                        Console.WriteLine("Confusion Matrix:");
                        Console.WriteLine($"    TP: {truePositives}, FN: {falseNegatives}");
                        Console.WriteLine($"    FP: {falsePositives}, TN: {trueNegatives}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error training or evaluating model '{name}': {ex.Message}");
                }
            }
        }
    }
}
