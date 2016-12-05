require 'torch'
require 'hdf5'
require 'image'
local utils = require 'utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.batch_size, 'Must provide batch size')

  self.task = opt.task

  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  
  self.image_paths = {
    train = '/train2014/images',
    val = '/val2014/images',
  }
  
  local train_size = self.h5_file:read(self.image_paths.train):dataspaceSize()
  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val):dataspaceSize()[1],
  }
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]

  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end
  
  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end
end


function DataLoader:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader:getBatch(split)
  local path = self.image_paths[split]

  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  
  -- Load images out of the HDF5 file
  local images = self.h5_file:read(path):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)

  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end
  local y = torch.Tensor(images:size(1),1,self.image_height,self.image_width)
  local scaled_y = torch.Tensor(images:size(1),1, 224, 224)
  
  local uv = torch.Tensor(images:size(1),2,self.image_height,self.image_width)
  
  -- YUV mean std
  local meanstd = {
       mean = { 0.485, 0.456, 0.406 },
       std = { 0.229, 0.224, 0.225 },
  }
  --[[
        RGB : 0 ~ 1
    
        Y : 0 ~ 1            -> -1 ~ 1
        U : -0.436 ~ 0.436   -> -1 ~ 1
        V : -0.615 ~ 0.615   -> -1 ~ 1
  ]]--
  for t=1,images:size(1) do
    local origin_img = image.scale(images[t], 256, 256)
    local scaled_img = image.scale(images[t], 224, 224)
    
    local yuv = image.rgb2yuv(origin_img)
    local scaled_yuv = image.rgb2yuv(scaled_img)
        
    y[t][1] = torch.add(yuv[1], -0.5):mul(2)
    scaled_y[t][1] = torch.add(scaled_yuv[1], -0.5):mul(2)
    uv[t][1] = torch.div(yuv[2], 0.436)
    uv[t][2] = torch.div(yuv[3], 0.615)
  end
  
  
  return y, scaled_y, uv
end
