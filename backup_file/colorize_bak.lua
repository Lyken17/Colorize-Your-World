--[[
Deploy the model...
--]]
require 'torch'
require 'nn'
require 'image'
require 'nngraph'

local utils = require 'utils'


--[[
Use a trained feedforward model to stylize either a single image or an entire
directory of images.
--]]

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'checkpoint.t7')

-- Input / output options
cmd:option('-input_image', '')
cmd:option('-output_image', 'out.png')
cmd:option('-input_dir', '')
cmd:option('-output_dir', '')

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 0)


local function main()
  local opt = cmd:parse(arg)

  if (opt.input_image == '') and (opt.input_dir == '') then
    error('Must give exactly one of -input_image or -input_dir')
  end

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local ok, checkpoint = pcall(function() return torch.load(opt.model) end)
  if not ok then
    print('ERROR: Could not load model from ' .. opt.model)
    print('You may need to download the pretrained models by running')
    print('bash download_colorization_model.sh')
    return
  end
  print('step 0')
  local model = checkpoint.model
  model:evaluate()
  model:type(dtype)

   print('step 0.5')
  local function run_image(in_path, out_path)

    local img = image.load(in_path)
    local H, W = img:size(2), img:size(3)
    local img = image.scale(img, torch.round(img:size(3)/8)*8, torch.round(img:size(2)/8)*8)
    print('step 0.75')
    local scaled_img = image.scale(img, 224, 224)
    H, W = img:size(2), img:size(3)

    print('step 1')
    if img:size(1) > 1 then
      img = image.rgb2y(img)
      scaled_img = image.rgb2y(scaled_img)
    end

    local img_pre = img:view(1, 1, H, W):type(dtype)
    local scaled_img_pre = scaled_img:view(1, 1, 224, 224):type(dtype)
    print('step 2')

    img_pre = torch.add(img_pre,-0.5)
    scaled_img = torch.add(scaled_img,-0.5)

    local input = {img_pre, scaled_img}
    local uv = model:forward(input)
    print('step 3')

    uv = uv:type('torch.DoubleTensor'):view(2,uv:size(3),uv:size(4))
    img_pre = img_pre:type('torch.DoubleTensor'):view(1,H,W)

    uv = image.scale(uv,W,H)
    local img_out = torch.cat(img_pre,uv,1)
    img_out = image.yuv2rgb(img_out)

    print('Writing output image to ' .. out_path)
    local out_dir = paths.dirname(out_path)
    if not path.isdir(out_dir) then
      paths.mkdir(out_dir)
    end
    image.save(out_path, img_out)
  end


  if opt.input_dir ~= '' then
    if opt.output_dir == '' then
      error('Must give -output_dir with -input_dir')
    end
    for fn in paths.files(opt.input_dir) do
      if utils.is_image_file(fn) then
        local in_path = paths.concat(opt.input_dir, fn)
        local out_path = paths.concat(opt.output_dir, fn)
        run_image(in_path, out_path)
      end
    end
  elseif opt.input_image ~= '' then
    if opt.output_image == '' then
      error('Must give -output_image with -input_image')
    end

    run_image(opt.input_image, opt.output_image)
  end
end


main()
