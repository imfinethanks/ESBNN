import os
import os.path as osp
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator,  LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.ops as ops
    import nvidia.dali.fn as fn
    import nvidia.dali.ops.random as random
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    '''
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    '''                                 
    images, labels = fn.readers.mxnet(path = osp.join(data_dir, "train.rec" if is_training else "val.rec"), 
                                    index_path = osp.join(data_dir, "train.idx" if is_training else "val.idx"),
                                    random_shuffle = is_training, 
                                    shard_id = shard_id if is_training else 0, 
                                    num_shards = num_shards if is_training else 1,
                                    pad_last_batch=True,
                                    name="Reader"
                                    )
                                    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels
            
def imagenet_loader_dali(args):
  
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    crop_size = 224
    val_size = 256

    gpu = args.gpu if args.gpu is not None else 0


    #pipe = HybridTrainPipe(batch_size=opt.batch_size, num_threads=opt.workers, device_id=int(gpu), data_dir=traindir, crop=crop_size, dali_cpu=opt.dali_cpu)
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipe.build()
    trainloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.FILL, fill_last_batch = True)

    #pipe = HybridValPipe(batch_size=opt.batch_size, num_threads=opt.workers, device_id=int(gpu), data_dir=valdir, crop=crop_size, size=val_size)
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipe.build()
    testloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
  
    return trainloader, testloader    
