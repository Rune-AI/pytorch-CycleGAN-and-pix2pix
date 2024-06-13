import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

def ExportToOnnx(opt):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    

    onnx_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, f"{opt.name}.onnx")

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)
        torch.onnx.export(
        model.netG.module,
        model.real_A,
        onnx_file,
        export_params = True,
        opset_version = 10,
        do_constant_folding = True,
        input_names = ['modelInput'],
        output_names = ['modelOutput'],
        dynamic_axes = {
            'modelInput' : {0 : 'batch_size'},
            'modelOutput' : {0 : 'batch_size'}
        }
        )
        break
    print("Exported model to {}".format(onnx_file))

def ExportToOnnxDynamic(opt):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    onnx_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, f"{opt.name}.onnx")

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)
        torch.onnx.dynamo_export(
            model.netG.module,
            model.real_A
            ).save(onnx_file)
        break
    print("Exported model to {}".format(onnx_file))

def TestOnnx(opt):
    import onnx
    from util import html
    from util.visualizer import save_images


    onnx_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, f"{opt.name}.onnx")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)


    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}_ONNX'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s_ONNX, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))



    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_file)

    dataset = create_dataset(opt)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        AtoB = opt.direction == 'AtoB'
        real_A = data['A' if AtoB else 'B']
        real_B = data['B' if AtoB else 'A']
        image_paths = data['A_paths' if AtoB else 'B_paths']

        outputs = ort_sess.run(None, {'modelInput': real_A.numpy()})
        
        visuals = {'real_A': real_A, 'fake_B': torch.tensor(outputs[0]), 'real_B': real_B}
        save_images(webpage, visuals, image_paths, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

def TestOriginal(opt):
    from util import html
    from util.visualizer import save_images

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

def TestCombined(opt):
    import onnx
    from util import html
    from util.visualizer import save_images


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model_no_eval = create_model(opt)      # create a model given opt.model and other options
    model_no_eval.setup(opt)               # regular setup: load and print networks; create schedulers
    

    onnx_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, f"{opt.name}.onnx")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_file)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        model_no_eval.set_input(data)  # unpack data from data loader
        model_no_eval.test()           # run inference


        AtoB = opt.direction == 'AtoB'
        real_A = data['A' if AtoB else 'B']
        real_B = data['B' if AtoB else 'A']
        image_paths = data['A_paths' if AtoB else 'B_paths']
        outputs = ort_sess.run(None, {'modelInput': real_A.numpy()})
        
        visuals = {'real_A': real_A, 'fake_B': model_no_eval.fake_B, 'fake_B_eval': model.fake_B, 'fake_B_ONNX': torch.tensor(outputs[0]), 'real_B': real_B}

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, image_paths))
        save_images(webpage, visuals, image_paths, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

def TestSingle(opt, imagepth):
    import onnx
    from util import html
    from util.visualizer import save_images
    from PIL import Image
    import torchvision.transforms as transforms
    from data.base_dataset import get_transform, get_params

    opt.load_size = 512

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    onnx_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, f"{opt.name}.onnx")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_file)

    # create a website
    web_dir = os.path.join(opt.results_dir, f"{opt.name}_Single", '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s_Single, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


    image = Image.open(imagepth).convert('RGB')
    transform_params = get_params(opt, image.size)
    transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
    image = transform(image).unsqueeze(0)

    model.real_A = image.to(model.device)
    model.real_B = image.to(model.device)

    model.test()           # run inference

    outputs = ort_sess.run(None, {'modelInput': image.numpy()})
    
    visuals = {'real_A': image, 'fake_B': model.fake_B, 'fake_B_ONNX': torch.tensor(outputs[0])}

    save_images(webpage, visuals, [""], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True
    ExportToOnnx(opt)
    # ExportToOnnxDynamic(opt)
    # TestOriginal(opt)
    # TestOnnx(opt)
    TestCombined(opt)
    # TestSingle(opt, "D:\DAYIII\LandscapeErosion\Saved\Debug\InputErosion.bmp") # Used for testing on exported unreal landscapes
    
