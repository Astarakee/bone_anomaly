import os
import SimpleITK as sitk

def segment(input_file, output_file, speed_func, curvature_weight = 0.5):
    command_args_template = "../mialab/MiaLab.sh -op ThresholdLevelset -CurvatureWeighting {} -QuitAfterDone 1 ".format(curvature_weight)
    command_args_template += "-SeedRegionFile FullVolume "
    command_args_template += "-lut:IntensityToSpeedLUT "
    first_node = True
    temp_file = "temp.mhd"
    for node in speed_func:
        if not first_node:
            command_args_template += ":"
        first_node = False
        command_args_template += "{}:{}".format(node[0], node[1])
    command_args_template += " "

    command_str = command_args_template + "-InputImage " + input_file + " -OutputImage " + temp_file
    print(command_str)
    os.system(command_str)
    assert os.path.isfile(temp_file)

    image = sitk.ReadImage(temp_file)
    mask = sitk.BinaryThreshold(image,0,100,255,0)
    sitk.WriteImage(mask, output_file)


def level_set_segmentation(input, init_seg, lower_threshold = 0.0, upper_threshold = 1.0, curvature_weight = 0.5):
    img = sitk.GetImageFromArray(input)
    seg = sitk.GetImageFromArray(init_seg)

    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower_threshold)
    lsFilter.SetUpperThreshold(upper_threshold)
    lsFilter.SetMaximumRMSError(0.02)
    lsFilter.SetNumberOfIterations(100)
    lsFilter.SetCurvatureScaling(curvature_weight)
    lsFilter.SetPropagationScaling(1)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(img, sitk.sitkFloat32))
    output = sitk.GetArrayFromImage(ls)
    return output
