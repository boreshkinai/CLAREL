# This is the implementation of the following paper
CLAREL: Classification via Retrieval Loss for Zero-Shot Learning
Boris N. Oreshkin, Negar Rostamzadeh, Pedro O. Pinheiro, Christopher Pal; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, pp. 916-917

https://openaccess.thecvf.com/content_CVPRW_2020/html/w54/Oreshkin_CLAREL_Classification_via_Retrieval_Loss_for_Zero-Shot_Learning_CVPRW_2020_paper.html

please cite as 

@InProceedings{oreshkin2020clarel,

  author = {Oreshkin, Boris N. and Rostamzadeh, Negar and Pinheiro, Pedro O. and Pal, Christopher},
  
  title = {{CLAREL}: Classification via Retrieval Loss for Zero-Shot Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2020}
}

NV_GPU=0,1 nvidia-docker run -p 1234:8888 -p 1235:6006 -p 1236:6007 -p 1237:6008 -p 1238:6009 -v /mnt/scratch/boris:/mnt/scratch/boris -v /mnt/datasets/public/:/mnt/datasets/public/ -v /mnt/home/boris:/mnt/home/boris -t -d --name boris_zeroshot_exp boris_zeroshot
