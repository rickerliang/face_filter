require 'ffmpeg'
require 'paths'
require 'image'

local channels = {'r', 'g', 'b'}
local videoLength = 46144
local negExampleFactor = opt.negExampleFactor
local frameWidth = 96
local frameHeight = 96
local frameRate = 1


local function applyNormalize(posData, negData)
    for i,channel in ipairs(channels) do
        -- normalize each channel globally:
        local mean = {}
        local std = {}
        mean[i] = posData[{ {},i,{},{} }]:mean()
        std[i] = posData[{ {},i,{},{} }]:std()
        posData[{ {},i,{},{} }]:add(-mean[i])
        posData[{ {},i,{},{} }]:div(std[i])
    end
    print('|')
    
    for i,channel in ipairs(channels) do
        -- normalize each channel globally:
        local mean = {}
        local std = {}
        mean[i] = negData[{ {},i,{},{} }]:mean()
        std[i] = negData[{ {},i,{},{} }]:std()
        negData[{ {},i,{},{} }]:add(-mean[i])
        negData[{ {},i,{},{} }]:div(std[i])
    end
    print('|')
    for i,channel in ipairs(channels) do
        local posMean = posData[{ {},i }]:mean()
        local posStd = posData[{ {},i }]:std()
        
        local negMean = negData[{ {},i }]:mean()
        local negStd = negData[{ {},i }]:std()
        
        print('pos data, '..channel..'-channel, mean:               ' .. posMean)
        print('pos data, '..channel..'-channel, standard deviation: ' .. posStd)
        
        print('neg data, '..channel..'-channel, mean:                   ' .. negMean)
        print('neg data, '..channel..'-channel, standard deviation:     ' .. negStd)
    end
end


-- return positiveExample--positiveLabel and negativeExample--negativeLabel dataset, example x ch x w x h--example
local function generateData()
    local trainingDataPath = 'training_data'
    paths.mkdir(trainingDataPath);
    
    if paths.filep(paths.concat(trainingDataPath, 'posExample.t7'))
        and paths.filep(paths.concat(trainingDataPath, 'posLabel.t7'))
        and paths.filep(paths.concat(trainingDataPath, 'negExample.t7'))
        and paths.filep(paths.concat(trainingDataPath, 'negLabel.t7')) then
        local posExample = torch.load(paths.concat(trainingDataPath, 'posExample.t7'))
        local posLabel = torch.load(paths.concat(trainingDataPath, 'posLabel.t7'))
        local negExample = torch.load(paths.concat(trainingDataPath, 'negExample.t7'))
        local negLabel = torch.load(paths.concat(trainingDataPath, 'negLabel.t7'))
        print('generated data found!')
        print('one example like')
        print(posExample[1]:size())
        print(posExample[{ 1,1,{},{} }]:mean())
        print(posExample[{ 1,1,{},{} }]:std())
        --image.display{image=posExample[1], legend='Some testing examples'}
        return posExample, posLabel, negExample, negLabel
    else
        print('generating data...')
        local positiveStreamPath = paths.concat('../favorite_face_classifier_data_set', 'pos')
        local negativeStreamPath = paths.concat('../favorite_face_classifier_data_set', 'neg')
        
        local posFileIter = paths.iterfiles(positiveStreamPath)
        local negFileIter = paths.iterfiles(negativeStreamPath)
        local exampleCount = videoLength * frameRate
        
        local posExample = torch.Tensor(exampleCount, #channels, frameWidth, frameHeight):zero()
        local posLabel = torch.Tensor(exampleCount):fill(1)
        local negExample = torch.Tensor(exampleCount * negExampleFactor, #channels, frameWidth, frameHeight):zero()
        local negLabel = torch.Tensor(exampleCount * negExampleFactor):fill(2)
        local shuffle = torch.randperm(exampleCount)
        local negShuffle = torch.randperm(exampleCount * negExampleFactor)
        if opt.normalizeSample == 1 then
            print('generating normalized data set ...')
            for i = 1, exampleCount do
                local imageFile = paths.concat(positiveStreamPath, posFileIter())
                local status, imageTensor = pcall(image.load, imageFile, #channels)
                if status ~= true then
                    print('load image error ' .. imageTensor)
                    print(imageFile)
                    assert(false)
                end
                posExample[shuffle[i]] =
                    normalizeRGB(image.scale(imageTensor, frameWidth, frameHeight))
                if i % 2000 == 0 then
                    io.write('*')
                    io.flush ()
                end
            end
            for j = 1, exampleCount * negExampleFactor do
                local imageFile = paths.concat(negativeStreamPath, negFileIter())
                local status, imageTensor = pcall(image.load, imageFile, #channels)
                if status ~= true then
                    print('load image error ' .. imageTensor)
                    print(imageFile)
                    assert(false)
                end
                negExample[negShuffle[j]] =
                    normalizeRGB(image.scale(imageTensor, frameWidth, frameHeight))
                if j % 2000 == 0 then
                    io.write('*')
                    io.flush ()
                end
            end
        else
            print('generating unnormalized data set ...')
            for i = 1, exampleCount do
                local imageFile = paths.concat(positiveStreamPath, posFileIter())
                local status, imageTensor = pcall(image.load, imageFile, #channels)
                if status ~= true then
                    print('load image error ' .. imageTensor)
                    print(imageFile)
                    assert(false)
                end
                posExample[shuffle[i]] =
                    image.scale(imageTensor, frameWidth, frameHeight)
                if i % 2000 == 0 then
                    io.write('*')
                    io.flush ()
                end
            end
            for j = 1, exampleCount * negExampleFactor do
                local imageFile = paths.concat(negativeStreamPath, negFileIter())
                local status, imageTensor = pcall(image.load, imageFile, #channels)
                if status ~= true then
                    print('load image error ' .. imageTensor)
                    print(imageFile)
                    assert(false)
                end
                negExample[negShuffle[j]] =
                    image.scale(imageTensor, frameWidth, frameHeight)
                if j % 2000 == 0 then
                    io.write('*')
                    io.flush ()
                end
            end
        end
        
        --applyNormalize(posExample, negExample)
        print('')
        print('save data to ' .. trainingDataPath)
        torch.save(paths.concat(trainingDataPath, 'posExample.t7'), posExample)
        torch.save(paths.concat(trainingDataPath, 'negExample.t7'), negExample)
        torch.save(paths.concat(trainingDataPath, 'posLabel.t7'), posLabel)
        torch.save(paths.concat(trainingDataPath, 'negLabel.t7'), negLabel)
        print('done')
        print('one example like')
        print(posExample[1]:size())
        print(posExample[{ 1,1,{},{} }]:mean())
        print(posExample[{ 1,1,{},{} }]:std())
        --image.display{image=posExample[1], legend='Some testing examples'}
        return posExample, posLabel, negExample, negLabel
    end
end

function getTrainingData()
    local posExample, posLabel, negExample, negLabel = generateData()
    local posExampleCount = posExample:size(1)
    local negExampleCount = negExample:size(1)
    assert(posExampleCount * negExampleFactor == negExampleCount)
    if opt.smallTest then
        posExampleCount = 1000
        negExampleCount = 1000
        negExampleFactor = 1
    end
    
    local testPosExampleCount = math.floor(posExampleCount * opt.patches)
    local trainPosExampleCount = posExampleCount - testPosExampleCount
    local testNegExampleCount = testPosExampleCount * negExampleFactor
    local trainNegExampleCount = trainPosExampleCount * negExampleFactor
    print('negative example factor : ' .. negExampleFactor)
    print('test data / all data : ' .. opt.patches)
    print('positive example count : ' .. posExampleCount)
    print('negative example count : ' .. negExampleCount)
    print('training positive example count : ' .. trainPosExampleCount)
    print('test positive example count : ' .. testPosExampleCount)
    print('training negative example count : ' .. trainNegExampleCount)
    print('test negative example count : ' .. testNegExampleCount)
    
    local testExampleCount = testPosExampleCount + testNegExampleCount
    local trainExampleCount = trainPosExampleCount + trainNegExampleCount
    
    local trainData = {
        data = torch.Tensor(trainExampleCount, #channels, frameHeight, frameWidth):zero(),
        testData = torch.Tensor(testExampleCount, #channels, frameHeight, frameWidth):zero(),
        labels = torch.Tensor(trainExampleCount):zero(),
        testLabels = torch.Tensor(testExampleCount):zero(),
        size = function() return trainExampleCount end,
        testSize = function() return testExampleCount end
    }
    
    local dataIndex = 1
    local posIndex = 1
    local negIndex = 0
    while dataIndex <= trainExampleCount do
        trainData.data[dataIndex] = posExample[posIndex]
        trainData.labels[dataIndex] = posLabel[posIndex]
        posIndex = posIndex + 1
        
        for i = 1, negExampleFactor do
            trainData.data[dataIndex + i] = negExample[negIndex + i]
            trainData.labels[dataIndex + i] = negLabel[negIndex + i]
        end
        negIndex = negIndex + negExampleFactor
        
        dataIndex = dataIndex + negExampleFactor + 1
    end

    print('training data positive example count : ' .. posIndex - 1)
    print('training data negative example count : ' .. negIndex)
    assert(posIndex - 1 == trainPosExampleCount)
    assert(negIndex == trainNegExampleCount)
    
    dataIndex = 1
    while dataIndex <= testExampleCount do
        trainData.testData[dataIndex] = posExample[posIndex]
        trainData.testLabels[dataIndex] = posLabel[posIndex]
        posIndex = posIndex + 1
        
        for i = 1, negExampleFactor do
            trainData.testData[dataIndex + i] = negExample[negIndex + i]
            trainData.testLabels[dataIndex + i] = negLabel[negIndex + i]
        end
        negIndex = negIndex + negExampleFactor
        
        dataIndex = dataIndex + negExampleFactor + 1
    end
    
    assert(posIndex - 1 == trainPosExampleCount + testPosExampleCount)
    assert(negIndex == trainNegExampleCount + testNegExampleCount)
    
    for l = 1, trainExampleCount do
        if trainData.labels[l] == 0 then
            print('valid training data fail at : ' .. l .. ' total : ' .. trainExampleCount)
            assert(false)
        end
    end
    
    for k = 1, testExampleCount do
        if trainData.testLabels[k] == 0 then
            print('valid test data fail at : ' .. k .. ' total : ' .. testExampleCount)
            assert(false)
        end
    end
    
    return trainData
end
