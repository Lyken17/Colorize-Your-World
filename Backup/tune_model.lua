require 'nn'
require 'nngraph'


local M = {}



function M.build_model(opts)
    opt = opts
    
    if opt.fine_tune then
        local d        = torch.load( 'colornet.t7' )
        local datamean = d.mean
        pre_trained_model    = d.model:float()
    end
    
    -- Convolution = cudnn.SpatialConvolution
    Convolution = nn.SpatialConvolution

    -- Avg = cudnn.SpatialAveragePooling
    Avg = nn.SpatialAveragePooling

    -- ReLU = cudnn.ReLU
    ReLU = nn.ReLU

    Max = nn.SpatialMaxPooling
    
    SBatchNorm = nn.SpatialBatchNormalization


    function vgg()
        print('initialize low level feature layer')
        if opt.fine_tune then
            print('loading weights for low level feature')
            local model = pre_trained_model.forwardnodes[4].data.module:clone()
            model.train = false
            return model
        else
            local model = nn.Sequential()
            model:add(Convolution(  1,  64, 3,3, 2,2, 1,1))
            model:add(ReLU(true))
            model:add(Convolution( 64, 128, 3,3, 1,1, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(128, 128, 3,3, 2,2, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(128, 256, 3,3, 1,1, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(256, 256, 3,3, 2,2, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(256, 512, 3,3, 1,1, 1,1))
            model:add(ReLU(true))
            return model
        end
    end

    function global_feature()
        print('initialize mid global feature layer')
        if opt.fine_tune then
            print('loading weights for global  feature')
            
            local model = pre_trained_model.forwardnodes[8].data.module:clone()
            model.train = false
            return model
        else
            local model = nn.Sequential()
            model:add(Convolution(512, 512, 3,3, 2,2, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(512, 512, 3,3, 1,1, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(512, 512, 3,3, 2,2, 1,1))
            model:add(ReLU(true))
            model:add(Convolution(512, 512, 3,3, 1,1, 1,1))
            model:add(ReLU(true))
            model:add(nn.View(-1, 25088))
            model:add(nn.Linear(25088, 1024))
            model:add(ReLU(true))
            model:add(nn.Linear(1024, 512))
            model:add(ReLU(true))
            return model
        end
    end

    function classification(nLabels)
        -- not used
        model = nn.Sequential()
        model:add(nn.Linear(512, 512))
        model:add(ReLU(true))
        model:add(nn.Dropout(0.5))

        model:add(nn.Linear(512, nLabels))
        model:add(nn.LogSoftMax())
        return model
    end

    function mid_level_feature()
        print('initialize mid level feature layer')
        if opt.fine_tune then
            print('loading weights for mid level feature')
            
            local model = pre_trained_model.forwardnodes[5].data.module:clone()
            model.train = false
            return model
        else
            local model = nn.Sequential()
            model:add(Convolution(512, 512, 3, 3, 1, 1, 1, 1))
            model:add(ReLU(true))
            model:add(Convolution(512, 256, 3, 3, 1, 1, 1, 1))
            model:add(ReLU(true))
            return model
        end
    end

    function img2feat(h, w)
        h_feat = h / 8
        w_feat = w / 8
        print('initialize fusion layer')
        if opt.fine_tune then
            local model = nn.Sequential()
            local pre_trained_linear = pre_trained_model.forwardnodes[9].data.module:get(1):clone()
            model:add(pre_trained_linear)
            model:add(ReLU(true))
            model:add(nn.Replicate(h_feat, 2, 1))
            model:add(nn.Replicate(w_feat, 2, 2))
            model.train = false
            return model
        else
            local model = nn.Sequential()
            model:add(nn.Linear(512, 256))
            model:add(ReLU(true))
            model:add(nn.Replicate(h_feat, 2, 1))
            model:add(nn.Replicate(w_feat, 2, 2))
            return model
        end
    end

    local function upsample_and_color()
        local model = nn.Sequential()

        model:add(Convolution(512, 256, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(Convolution(256, 128, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(nn.SpatialUpSamplingNearest(2))

        model:add(Convolution(128, 64, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(Convolution( 64, 64, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(nn.SpatialUpSamplingNearest(2))

        model:add(Convolution( 64, 32, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(Convolution( 32,  16, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(nn.SpatialUpSamplingNearest(2))

        model:add(Convolution( 16, 8, 3,3, 1,1, 1,1))
        model:add(ReLU(true))
        model:add(Convolution( 8,  2, 3,3, 1,1, 1,1))
        model:add(nn.Tanh())
        return model
    end
    
    h = 256
    w = 256
    origin_img = nn.Identity()()
    scaled_img = nn.Identity()()

    -- local feature / original image
    local feature = nn.MapTable(vgg())({origin_img, scaled_img})

    local low_feat1 = nn.SelectTable(1)(feature)
    local low_feat2 = nn.SelectTable(2)(feature)

    local mid_feat = mid_level_feature()(low_feat1)
    local global_feat = global_feature()(low_feat2)
    local fusion_layer = img2feat(h, w)(global_feat)


    local mixed_res = nn.JoinTable(2)({mid_feat, fusion_layer})

    local predict_UV = upsample_and_color()(mixed_res)

    net = nn.gModule({origin_img, scaled_img}, {predict_UV})
    return net
end

return M

