function diabetesModelAnalysis()
    % DIABETESMODELANALYSIS - Main function for diabetes prediction analysis
    % Performs comprehensive model evaluation including learning curves and
    % bias-variance analysis for diabetes prediction
    clc; close all; 
    % Load and preprocess data
    [X, Y, featureNames] = loadDiabetesData();
    
    % Define models to evaluate
    models = defineModels();
    
    % Set up stratified 10-fold cross-validation
    k = 10;
    cv = cvpartition(Y, 'KFold', k, 'Stratify', true);
    
    % Generate and plot learning curves
    plotLearningCurves(X, Y, cv, models, featureNames);
    
    % Perform and plot bias-variance analysis
    plotBiasVariance(X, Y, cv, models, featureNames);
end

function [X, Y, featureNames] = loadDiabetesData()
    % LOADDIABETESDATA - Loads and preprocesses diabetes dataset
    % Returns:
    %   X - Feature matrix (normalized)
    %   Y - Response vector (0=no diabetes, 1=diabetes)
    %   featureNames - Cell array of feature names
    
    % Load dataset from GitHub
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv';
    data = webread(url);
    dataLines = splitlines(data);
    numLines = length(dataLines);
    data = zeros(numLines, 9); % 8 features + 1 outcome
    
    % Parse data lines
    for i = 1:numLines
        if ~isempty(dataLines{i})
            lineData = str2double(strsplit(dataLines{i}, ','));
            if numel(lineData) == 9
                data(i,:) = lineData;
            end
        end
    end
    
    % Remove empty lines and create feature matrix
    data = data(~isnan(data(:,1)),:);
    X = data(:,1:8);
    Y = data(:,9);
    
    % Handle missing values (zeros)
    X(X==0) = NaN;
    for i = 1:size(X,2)
        colData = X(:,i);
        colData(isnan(colData)) = median(colData(~isnan(colData)),'omitnan');
        X(:,i) = colData;
    end
    
    % Feature engineering
    X(:,9) = X(:,2).*X(:,6); % Glucose-BMI interaction
    X(:,10) = X(:,5)./X(:,6); % Insulin-to-BMI ratio
    
    % Feature names
    featureNames = {'Pregnancies','Glucose','BP','SkinThickness',...
                   'Insulin','BMI','DPF','Age','Glucose_BMI','Insulin_BMI'};
    
    % Normalize features
    X = normalize(X);
end

function models = defineModels()
    % DEFINEMODELS - Returns cell array of classification models to evaluate
    % Each model is defined as {name, function_handle}
    
    models = {
        % Decision Tree with limited splits
        'Decision Tree', @(X,Y) fitctree(X,Y,'MaxNumSplits',20,'MinParentSize',10);
        
        % Logistic Regression with ridge regularization
        'Logistic Regression', @(X,Y) fitclinear(X,Y,'Learner','logistic',...
                              'Regularization','ridge','Lambda',0.1);
        
        % SVM with RBF kernel
        'SVM', @(X,Y) fitcsvm(X,Y,'KernelFunction','rbf',...
                  'Standardize',true,'BoxConstraint',1);
        
        % Random Forest with 50 trees
        'Random Forest', @(X,Y) TreeBagger(50,X,Y,'Method','classification',...
                          'OOBPredictorImportance','on');
        
        % Bagged Trees with 30 trees
        'Bagged Trees', @(X,Y) TreeBagger(30,X,Y,'Method','classification',...
                         'OOBPredictorImportance','on');
        
        % Naive Bayes with kernel distributions
        'Naive Bayes', @(X,Y) fitcnb(X,Y,'DistributionNames','kernel');
        
        % XGBoost using LogitBoost with learning rate 0.1
        'XGBoost', @(X,Y) fitcensemble(X,Y,'Method','LogitBoost',...
                   'Learners',templateTree('MaxNumSplits',10),'LearnRate',0.1);
    };
end

function plotLearningCurves(X, Y, cv, models, featureNames)
    % PLOTLEARNINGCURVES - Generates and plots learning curves for all models
    
    % Set up figure
    figure('Name','Learning Curves','Position',[100 100 1200 800], 'Color','w');
    numModels = length(models);
    colors = lines(numModels);
    
    % Define training set sizes (logarithmic scale)
    num_samples = size(X,1);
    train_sizes = unique(round(logspace(log10(20),log10(num_samples),10)));
    train_sizes = train_sizes(train_sizes < num_samples*0.9); % Leave some for validation
    
    % Initialize results storage
    train_scores = cell(numModels,1);
    val_scores = cell(numModels,1);
    
    % For each model
    for m = 1:numModels
        fprintf('Generating learning curve for %s...\n', models{m,1});
        
        % Initialize for this model
        train_scores{m} = zeros(length(train_sizes), cv.NumTestSets);
        val_scores{m} = zeros(length(train_sizes), cv.NumTestSets);
        
        % For each training set size
        for s = 1:length(train_sizes)
            current_size = train_sizes(s);
            
            % For each fold
            for fold = 1:cv.NumTestSets
                % Get indices for this fold
                trainIdx = training(cv,fold);
                testIdx = test(cv,fold);
                
                % Subsample training data
                subIdx = randperm(sum(trainIdx), min(current_size, sum(trainIdx)));
                X_train = X(trainIdx,:);
                Y_train = Y(trainIdx);
                X_train = X_train(subIdx,:);
                Y_train = Y_train(subIdx);
                X_test = X(testIdx,:);
                Y_test = Y(testIdx);
                
                try
                    % Train model
                    model = models{m,2}(X_train, Y_train);
                    
                    % Get predictions (handling different model types)
                    if strcmp(models{m,1}, 'Logistic Regression')
                        [~, train_score] = predict(model, X_train);
                        [~, val_score] = predict(model, X_test);
                    else
                        [~, train_score] = predict(model, X_train);
                        [~, val_score] = predict(model, X_test);
                    end
                    
                    % Store AUC scores
                    [~,~,~,train_scores{m}(s,fold)] = perfcurve(Y_train, train_score(:,2), 1);
                    [~,~,~,val_scores{m}(s,fold)] = perfcurve(Y_test, val_score(:,2), 1);
                    
                catch ME
                    fprintf('Error in %s (size=%d, fold=%d): %s\n',...
                            models{m,1}, current_size, fold, ME.message);
                    train_scores{m}(s,fold) = NaN;
                    val_scores{m}(s,fold) = NaN;
                end
            end
        end
        
        % Plotting for this model
        subplot(2, ceil(numModels/2), m);
        hold on; grid on;
        
        % Calculate mean and std
        mean_train = mean(train_scores{m}, 2, 'omitnan');
        std_train = std(train_scores{m}, 0, 2, 'omitnan');
        mean_val = mean(val_scores{m}, 2, 'omitnan');
        std_val = std(val_scores{m}, 0, 2, 'omitnan');
        
        % Plot training scores with error bands
        errorbar(train_sizes, mean_train, std_train, 'Color', colors(m,:), ...
                'LineWidth', 2, 'CapSize', 0);
        plot(train_sizes, mean_train, 'o', 'Color', colors(m,:), ...
            'MarkerFaceColor', colors(m,:), 'MarkerSize', 6);
        
        % Plot validation scores with error bands
        errorbar(train_sizes, mean_val, std_val, 'Color', colors(m,:), ...
                'LineStyle', '--', 'LineWidth', 2, 'CapSize', 0);
        plot(train_sizes, mean_val, 's', 'Color', colors(m,:), ...
            'MarkerFaceColor', 'w', 'MarkerSize', 6);
        
        % Formatting
        title(models{m,1}, 'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Training Set Size', 'FontSize', 10);
        ylabel('AUC', 'FontSize', 10);
        legend({'Training', '', 'Validation', ''}, 'Location', 'southeast');
        xlim([min(train_sizes) max(train_sizes)]);
        ylim([0.5 1]);
        set(gca, 'FontSize', 9, 'LineWidth', 1.5);
    end
    
    sgtitle('Learning Curves (Mean AUC ± 1 STD)', 'FontSize', 14, 'FontWeight', 'bold');
end

function plotBiasVariance(X, Y, cv, models, featureNames)
    % PLOTBIASVARIANCE - Performs and plots bias-variance analysis
    
    % Set up figure
    figure('Name','Bias-Variance Analysis','Position',[100 100 1200 600], 'Color','w');
    numModels = length(models);
    colors = lines(numModels);
    
    % Initialize results storage
    bias = zeros(numModels,1);
    variance = zeros(numModels,1);
    total_error = zeros(numModels,1);
    
    % Generate predictions from all models
    all_preds = zeros(size(X,1), numModels);
    
    for m = 1:numModels
        % Initialize for this model
        model_preds = zeros(size(X,1), cv.NumTestSets);
        
        % Cross-validation predictions
        for fold = 1:cv.NumTestSets
            trainIdx = training(cv,fold);
            testIdx = test(cv,fold);
            
            try
                % Train model
                model = models{m,2}(X(trainIdx,:), Y(trainIdx));
                
                % Get predictions
                if strcmp(models{m,1}, 'Logistic Regression')
                    [~, scores] = predict(model, X(testIdx,:));
                else
                    [~, scores] = predict(model, X(testIdx,:));
                end
                
                model_preds(testIdx,fold) = scores(:,2);
            catch ME
                fprintf('Error in %s (fold %d): %s\n', models{m,1}, fold, ME.message);
                model_preds(testIdx,fold) = NaN;
            end
        end
        
        % Store mean predictions for this model
        all_preds(:,m) = mean(model_preds, 2, 'omitnan');
    end
    
    % Calculate bias and variance components
    avg_pred = mean(all_preds, 2); % Average predictions across models
    for m = 1:numModels
        % Bias: squared difference between average prediction and true labels
        bias(m) = mean((avg_pred - Y).^2);
        
        % Variance: variability of individual model predictions
        variance(m) = mean(var(all_preds, 0, 2));
        
        % Total error
        total_error(m) = mean((all_preds(:,m) - Y).^2);
    end
    
    % Normalize for visualization
    max_val = max([bias; variance; total_error]);
    bias = bias/max_val;
    variance = variance/max_val;
    total_error = total_error/max_val;
    
    % PLOT 1: Bias-Variance Decomposition
    subplot(1,2,1);
    h = bar([bias, variance, total_error-bias-variance], 'stacked');
    
    % Customize colors
    set(h(1), 'FaceColor', [0.8 0.2 0.2], 'DisplayName', 'Bias²');
    set(h(2), 'FaceColor', [0.2 0.2 0.8], 'DisplayName', 'Variance');
    set(h(3), 'FaceColor', [0.9 0.9 0.9], 'DisplayName', 'Irreducible Error');
    
    % Formatting
    set(gca, 'XTick', 1:numModels, 'XTickLabel', {models{:,1}}, ...
        'XTickLabelRotation', 45, 'FontSize', 10, 'LineWidth', 1.5);
    ylabel('Normalized Error', 'FontSize', 11, 'FontWeight', 'bold');
    title('Bias-Variance Decomposition', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'northoutside', 'Orientation', 'horizontal');
    grid on;
    box off;
    
    % PLOT 2: Bias-Variance Tradeoff
    subplot(1,2,2);
    hold on;
    grid on;
    
    % Plot tradeoff points
    h = scatter(bias, variance, 150, colors, 'filled');
    
    % Add model labels
    for m = 1:numModels
        text(bias(m), variance(m), models{m,1}, ...
            'FontSize', 9, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'Color', colors(m,:));
    end
    
    % Plot ideal tradeoff line
    plot([0 1], [1 0], 'k--', 'LineWidth', 1.5);
    text(0.5, 0.5, 'Ideal Tradeoff', 'Rotation', -45, ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Formatting
    xlabel('Bias²', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Variance', 'FontSize', 11, 'FontWeight', 'bold');
    title('Bias-Variance Tradeoff', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0 1]);
    ylim([0 1]);
    axis square;
    set(gca, 'FontSize', 10, 'LineWidth', 1.5);
    
    % Overall title
    sgtitle('Model Bias-Variance Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

