require 'ffmpeg'
require 'paths'
require 'image'

local channels = {'r', 'g', 'b'}
local videoLength = 33914
local frameWidth = 96
local frameHeight = 96
local frameRate = 1

local function normalizeRGB(image)
    for i,channel in ipairs(channels) do
        -- normalize each channel globally:
        local mean = image[{ i,{},{} }]:mean()
        local std = image[{ i,{},{} }]:std()
        image[{ i,{},{} }]:add(-mean)
        image[{ i,{},{} }]:div(std + 1e-8)
        --print('image data, '..channel..'-channel, mean:               ' .. image[{i, {}, {}}]:mean())
        --print('image data, '..channel..'-channel, standard deviation: ' .. image[{i, {}, {}}]:std())
    end
    return image
end

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
        --image.display{image=posExample[videoLength * frameRate], legend='Some pos examples'}
        --image.display{image=negExample[videoLength * frameRate], legend='Some neg examples'}
        return posExample, posLabel, negExample, negLabel
    else
        print('generating data...')
        local positiveExamplePath = paths.concat('../face_detector_data_set', 'pos')
        local negativeExamplePath = paths.concat('../face_detector_data_set', 'neg')
        
        local posFileIter = paths.iterfiles(positiveExamplePath)
        local negFileIter = paths.iterfiles(negativeExamplePath)
        local exampleCount = videoLength * frameRate
        
        local posExample = torch.Tensor(exampleCount, #channels, frameWidth, frameHeight)
        local posLabel = torch.Tensor(exampleCount):fill(1)
        local negExample = torch.Tensor(exampleCount, #channels, frameWidth, frameHeight)
        local negLabel = torch.Tensor(exampleCount):fill(2)
        local shuffle = torch.randperm(exampleCount)
        if opt.normalizeSample == 1 then
            print('generating normalized data set ...')
            for i = 1, exampleCount do
                local imageFile = paths.concat(positiveExamplePath, posFileIter())
                posExample[shuffle[i]] =
                    normalizeRGB(image.scale(image.load(imageFile, #channels), frameWidth, frameHeight))
                imageFile = paths.concat(negativeExamplePath, negFileIter())
                negExample[shuffle[i]] =
                    normalizeRGB(image.scale(image.load(imageFile, #channels), frameWidth, frameHeight))
                if i % 1000 == 0 then
                    print('|')
                end
            end
        else
            print('generating unnormalized data set ...')
            for i = 1, exampleCount do
                local imageFile = paths.concat(positiveExamplePath, posFileIter())
                posExample[shuffle[i]] =
                    image.scale(image.load(imageFile, #channels), frameWidth, frameHeight)
                imageFile = paths.concat(negativeExamplePath, negFileIter())
                negExample[shuffle[i]] =
                    image.scale(image.load(imageFile, #channels), frameWidth, frameHeight)
                if i % 1000 == 0 then
                    print('|')
                end
            end
        end
        
        --applyNormalize(posExample, negExample)
        
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
        --image.display{image=negExample[1], legend='Some testing examples'}
        return posExample, posLabel, negExample, negLabel
    end
end

function getTrainingData()
    local posExample, posLabel, negExample, negLabel = generateData()
    local posExampleCount = posExample:size(1)
    local negExampleCount = negExample:size(1)
    local exampleCount = posExampleCount * 2
    if opt.smallTest then
        posExampleCount = 1000
        exampleCount = 2000
    end
    local testExampleCount = math.floor(exampleCount * opt.patches)
    local trainExampleCount = exampleCount - testExampleCount
    print('pos example count : ' .. posExampleCount)
    print('neg example count : ' .. negExampleCount)
    print('total example pair : ' .. exampleCount)
    print('train example pair : ' .. trainExampleCount)
    print('test example pair : ' .. testExampleCount)
    local trainData = {
        data = torch.Tensor(trainExampleCount, #channels, frameHeight, frameWidth),
        testData = torch.Tensor(testExampleCount, #channels, frameHeight, frameWidth),
        labels = torch.Tensor(trainExampleCount),
        testLabels = torch.Tensor(testExampleCount),
        size = function() return trainExampleCount end,
        testSize = function() return testExampleCount end
    }
    
    local index = 1
    --local testPart = math.floor(posExampleCount * opt.patches)
    --local trainPart = posExampleCount - testPart
    local i = 1
    while index + 1 <= trainExampleCount  do
        trainData.data[index] = posExample[i]
        trainData.labels[index] = posLabel[i]
        trainData.data[index + 1] = negExample[i]
        trainData.labels[index + 1] = negLabel[i]
        index = index + 2
        i = i + 1
    end
    
    print('generate train data complete : ' .. i - 1)
    
    index = 1
    while index + 1 <= testExampleCount do
        trainData.testData[index] = posExample[i]
        trainData.testLabels[index] = posLabel[i]
        trainData.testData[index + 1] = negExample[i]
        trainData.testLabels[index + 1] = negLabel[i]
        index = index + 2
        i = i + 1
    end
    
    print('generate test data complete : ' .. i - 1)
    
    paths.rmall('scratch', 'yes')
    
    --print('save testData for debug ')
    --paths.mkdir('debug')
    --torch.save('debug/testData.t7', trainData.testData)
    --torch.save('debug/testLabel.t7', trainData.testLabels)
    
    return trainData
end

