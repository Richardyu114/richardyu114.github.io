const workboxVersion="5.1.3";importScripts("https://storage.googleapis.com/workbox-cdn/releases/5.1.3/workbox-sw.js"),workbox.core.setCacheNameDetails({prefix:"Densecollections"}),workbox.core.skipWaiting(),workbox.core.clientsClaim(),workbox.precaching.precacheAndRoute([{revision:"50ffbc82b75c0fc615992f85f18f322b",url:"./404.html"},{revision:"ba01ae7d9181747ab8e2613c9e36d208",url:"./about/index.html"},{revision:"b7c96d30be4f7b06df2dd679117ce31c",url:"./archives/2019/02/index.html"},{revision:"b1854d93059563a6286daf7aef1e886e",url:"./archives/2019/03/index.html"},{revision:"2273427724ce8de2339590ab0803d728",url:"./archives/2019/04/index.html"},{revision:"ad085c109b263afd1e2b48f9860d668f",url:"./archives/2019/05/index.html"},{revision:"9aaeacad4f1a189ffe683fa4122e5ab8",url:"./archives/2019/10/index.html"},{revision:"e24439486fed7e2c0a908204d0c03850",url:"./archives/2019/11/index.html"},{revision:"3b3d03631efdab797864b4ae4824e18e",url:"./archives/2019/index.html"},{revision:"bccab08f61154a874473ce3379a9c77e",url:"./archives/2019/page/2/index.html"},{revision:"b1d5701467b09f8d636b105629e768b3",url:"./archives/2020/01/index.html"},{revision:"5eb5ed0938c98a092d61809dd485081b",url:"./archives/2020/02/index.html"},{revision:"fa1abb1270705b82e53ac3d761b72c45",url:"./archives/2020/03/index.html"},{revision:"ec8070e5aaa8cfb52efe6bcff13c4d85",url:"./archives/2020/05/index.html"},{revision:"3f56e4419aab364500f0eb176b763432",url:"./archives/2020/06/index.html"},{revision:"0dc1b806d643d102687f84cd2e0fdf91",url:"./archives/2020/08/index.html"},{revision:"df75d9c793629099ced5bc324643841c",url:"./archives/2020/index.html"},{revision:"c9205e6c289c20dee1bbafeb8e94233e",url:"./archives/2020/page/2/index.html"},{revision:"4c050a03b52fd943246f7454e453d450",url:"./archives/2021/01/index.html"},{revision:"3665fc03b3f621c65e0f3cf465771945",url:"./archives/2021/index.html"},{revision:"33bc36337ed972c0eee898dfcd036a81",url:"./archives/index.html"},{revision:"ee6d80ca72f8c28a3b247626c72ad58f",url:"./archives/page/2/index.html"},{revision:"436f3dbdda366bcb0822b8aa7a6c35f7",url:"./archives/page/3/index.html"},{revision:"196e3e173cb7a8fe2416353768673e41",url:"./categories/博客搭建/index.html"},{revision:"20218d713667e7c5c24377454d95dacc",url:"./categories/工作总结/index.html"},{revision:"bcec54086dbb4575191cc562be9c8ce5",url:"./categories/基础数学/index.html"},{revision:"e400d493b0bb5c5ac90e0c4bef895b7c",url:"./categories/纪录与总结/index.html"},{revision:"d1431eb23eff8d5369e727aa0f668bb4",url:"./categories/技术支持/index.html"},{revision:"19fff83973ff767eb5c99331f8063bd5",url:"./categories/科研记录/index.html"},{revision:"f0af301e3a3fee2cf83060177750c7ae",url:"./categories/课程记录/index.html"},{revision:"d188b569b2a6569a5d331303fdb49e39",url:"./categories/论文阅读/index.html"},{revision:"9c8a63f2fbe622c0a51dd484ab939d1f",url:"./categories/深度学习环境配置/index.html"},{revision:"9b235d8b1f4642a6a0b22a9289c372eb",url:"./categories/算法与实现/index.html"},{revision:"a3d545cd67651dd8d6c5351727b1b9cf",url:"./categories/随笔杂谈/index.html"},{revision:"b33f30c6880fa06f08c2e8d7756cb8b9",url:"./categories/index.html"},{revision:"7b4458286ffe2ce32989c7723a4b09ad",url:"./categories/MissingcourseofCS/index.html"},{revision:"1357c2a6be69889517317f3b9a3df424",url:"./css/index.css"},{revision:"d41d8cd98f00b204e9800998ecf8427e",url:"./css/var.css"},{revision:"6870c1a44f4bc81d478ddec670b911c5",url:"./index.html"},{revision:"65d29fbdddb25f5ccc6d8e3364bf080d",url:"./js/main.js"},{revision:"d4ca891c702c547a356f6edbbfdc1e0f",url:"./js/search/algolia.js"},{revision:"de25bf0fd67bcc6c161201f99c8f7f29",url:"./js/search/local-search.js"},{revision:"4ba78e765ae6fe7bec46bb3c4f37b94e",url:"./js/tw_cn.js"},{revision:"5d617e1a33f31aa82474eb7f4b07717d",url:"./js/utils.js"},{revision:"8d0edfac8cfaf152390622dc8d0e2f50",url:"./leancloud_memo.json"},{revision:"29b599b8174b99be54430e350591f479",url:"./link/index.html"},{revision:"b9b2b455a3c4a36b4527e5d913a5fcdf",url:"./manifest.json"},{revision:"dc629de6678b97dbdd5cd8721b1f1928",url:"./MindWandering/index.html"},{revision:"f8b2a0cef382640e57606a2bec7f22d3",url:"./page/2/index.html"},{revision:"eb9985c8456dd75f56754850c47cc3ee",url:"./page/3/index.html"},{revision:"6c32cc5c79eac60c1b2707c2c99f43ca",url:"./PaperStation/index.html"},{revision:"92644fd05e3925f1199f8dbffb232820",url:"./posts/4074/index.html"},{revision:"12e4e4da03c4d531b56bee2f358723d9",url:"./posts/beyondsupervisedlearning/index.html"},{revision:"f214fb112508e939edceb3475e4f8ccc",url:"./posts/BriefreviewofObjectdetection/index.html"},{revision:"51261bab4547df50f4be5ec22598142f",url:"./posts/buildDLenvir/index.html"},{revision:"ae70fdbcc0e8bd5b7e146830b9d787f1",url:"./posts/controlLinuxserver/index.html"},{revision:"8b45b714a1b86a5ebb5c260a6b17d96a",url:"./posts/FutureMappingofAJD-1/index.html"},{revision:"b71ffd93230023cf46b2890fdb104891",url:"./posts/FutureMappingofAJD-2/index.html"},{revision:"62d21701441fbdd92552edbb7fb8beb1",url:"./posts/HexoBlogBulid/index.html"},{revision:"197995bcf3dd54abfeea6ab796bfcd9e",url:"./posts/houjieC++/index.html"},{revision:"797cdcb632e01db49a04476d660c7197",url:"./posts/houjieC++STL/index.html"},{revision:"64225502948539e3c338aa8d4e37fe22",url:"./posts/loopdetectioninBipartitegraph/index.html"},{revision:"45b29a017a9dc29f7ef416b8653960c1",url:"./posts/megviidlcourse/index.html"},{revision:"fa76c6191cc204f3a49708055208b998",url:"./posts/notesofCS231N/index.html"},{revision:"32a1136bebc81ad54e6a107de3d67b5d",url:"./posts/notesofLinearAlgebra/index.html"},{revision:"eee020d10c46e6c53b7fce523d47ac79",url:"./posts/personalthoughts-1/index.html"},{revision:"4e53b3088df61820eea4dbf7914a8742",url:"./posts/RCNNseries-1/index.html"},{revision:"a51403dfd69897cbb1a3b7211264c733",url:"./posts/RCNNseries-2/index.html"},{revision:"e2797181662e8dffeda28546187170aa",url:"./posts/ssd-refinedet-paper/index.html"},{revision:"f6386ffdc5de9fb2e37be5561d799263",url:"./posts/Summaryofthisyear-1/index.html"},{revision:"62a94e00387e601b00fcec64f1aea027",url:"./posts/Summaryofthisyear-2/index.html"},{revision:"0ea1d7009bf655241ed2840d5320ae4a",url:"./posts/thoughtsofintern/index.html"},{revision:"8d1ff72a1dff57047c496cc28f95d34d",url:"./posts/visualSLAMbyGaoxiang-1/index.html"},{revision:"629350744c4b47e08471eb27b33085a2",url:"./posts/visualSLAMbyGaoxiang-2/index.html"},{revision:"8b761c957fb53fb8248770cf20563c2b",url:"./posts/visualSLAMbyGaoxiang-3/index.html"},{revision:"d5c22a395a642e2f5b9674a99ced0620",url:"./posts/worksummaryofintern/index.html"},{revision:"1bd84829cc9f31dfe9513d25bfce2c30",url:"./posts/yolopaperreading/index.html"},{revision:"000711cfa6cd277eac72d523565b6cc2",url:"./tags/侯捷/index.html"},{revision:"461a9f8544cc2ff0d486f8e5786fa36a",url:"./tags/AI-system/index.html"},{revision:"9fdc692f24bcdcac312e457b347bcf80",url:"./tags/algorithm/index.html"},{revision:"69244eed39dcf7cdcf69096ba9797a85",url:"./tags/C-11/index.html"},{revision:"3192445107c4dad15e66965ab91e47ec",url:"./tags/CNN/index.html"},{revision:"1892408ea0e66207d2bbe560846d0010",url:"./tags/coding/index.html"},{revision:"eed8e4c86558e57af4ee8389fb5af7aa",url:"./tags/computer-science/index.html"},{revision:"b85431e477e0fe7115ab6cc61c82ede7",url:"./tags/computer-vision/index.html"},{revision:"c7e0ae114df32ede6e21776067ff99fa",url:"./tags/computer-vision/page/2/index.html"},{revision:"02b4c26c0360fe512da3ed58e8e311f3",url:"./tags/contest/index.html"},{revision:"5b500bf0f1135d9aadbd07a0805a7958",url:"./tags/deep-learning/index.html"},{revision:"aca043b56f2d2cefe758a7afea2d200d",url:"./tags/deep-neural-network/index.html"},{revision:"37877858a16ed4de96c06172aa45ef20",url:"./tags/domain-adaptation/index.html"},{revision:"4a0124727973c9d6a3f80863e6ebb45d",url:"./tags/Eigen/index.html"},{revision:"4081a6e8754429cb1219c832f2ab762b",url:"./tags/emotions/index.html"},{revision:"5bc420cc06a5ff634595c6bd7cec1a16",url:"./tags/feelings/index.html"},{revision:"edacff37210412f2042edccaaaf16521",url:"./tags/git/index.html"},{revision:"4d5524a8bbf4586afc63951068ccb53c",url:"./tags/growing/index.html"},{revision:"3674f79ac27b427d170b9fb71e4e382b",url:"./tags/hardware/index.html"},{revision:"2224eba0cb2ee1916f9860301b0386d0",url:"./tags/hexo/index.html"},{revision:"3f5d4215c6cfd4e83b5b6bf35999bd3b",url:"./tags/index.html"},{revision:"62ee016ac4c7d8f47d2c119ecbd9cc05",url:"./tags/job/index.html"},{revision:"11d6d559136ea37a19eb7b160823e707",url:"./tags/leancloud/index.html"},{revision:"cc3754650b13ee40af1855bc4cd95f72",url:"./tags/learning/index.html"},{revision:"45bd2e3a2f9a12a99b27d1156aac309c",url:"./tags/linear-algebra/index.html"},{revision:"425ea8f8885cab3f8d9c59daf7c8da8c",url:"./tags/linux/index.html"},{revision:"c40b6095587070b2d2a0fa486aba1262",url:"./tags/mapping/index.html"},{revision:"4bab2d887d3a366a326fe86a90f2208b",url:"./tags/mathematics/index.html"},{revision:"8b70e693d43394ce83f676c91a53892c",url:"./tags/medical-image-analysis/index.html"},{revision:"1c53b8cfa313d0527c14da5dbf912c53",url:"./tags/Megvii/index.html"},{revision:"3ec6ab0cec1bbc1ca6a062d973ce777d",url:"./tags/memory/index.html"},{revision:"a78b06e6a9501b2c1b6c75b03b49e6c1",url:"./tags/MIT/index.html"},{revision:"4b9e8d8b6c107b3a489fc0a135d710e3",url:"./tags/movies/index.html"},{revision:"b3f52565f185f5d263c8c449c2139572",url:"./tags/next/index.html"},{revision:"c1a6b8fc8a9e87727bb003eccc38bd58",url:"./tags/nvidia-driver/index.html"},{revision:"ec2bba7b910d60463d0d617aa31112c4",url:"./tags/object-detection/index.html"},{revision:"3b4ed1affc564b113d0f2fbf13bb8103",url:"./tags/OpenCV/index.html"},{revision:"a7e70f3b7f1ab01b55bfd484bacba32a",url:"./tags/practical-skills/index.html"},{revision:"0e5272473c4bbe01e9901ac2043b03cf",url:"./tags/public-course/index.html"},{revision:"d70ce9c32c1750839bff0beaac1b4363",url:"./tags/python/index.html"},{revision:"e85ce40797c9c04b47f71144bf75cb78",url:"./tags/reflection/index.html"},{revision:"c86eb4bc6e70f936128aec16277eb8a5",url:"./tags/reflections/index.html"},{revision:"66f8c177c41889e32b6447505d8b3e5a",url:"./tags/semantic-segmentation/index.html"},{revision:"355f25a395d5958ab9416cc9f09cfacf",url:"./tags/SLAM/index.html"},{revision:"b118637d4ac0d8273b190a98df4853d8",url:"./tags/Sophus/index.html"},{revision:"7397d61140f77f5d100f62bd37cc4b60",url:"./tags/ssh/index.html"},{revision:"1a02777c43fdadb37ddc1570b962cb32",url:"./tags/Standford/index.html"},{revision:"78f8b871bb5d8f4e42041a3c1d931c4c",url:"./tags/STL/index.html"},{revision:"d700c8ef736db87181f3d335b804a4ae",url:"./tags/summary/index.html"},{revision:"949eab4d9a46856acb1b39a4f42bdb3b",url:"./tags/supervised-learning/index.html"},{revision:"b266d3509a950c54aa7c80f5c382b739",url:"./tags/survey/index.html"},{revision:"7dfdee4b092d9bf06e739a4356c3d62b",url:"./tags/thoughts/index.html"},{revision:"69831ae68d38d4ea7d6c5ab7874dc34d",url:"./tags/Ubuntu/index.html"},{revision:"ef323a626ae9648585fbd2e05f7b40ca",url:"./tags/visual-SLAM/index.html"}],{directoryIndex:null}),workbox.precaching.cleanupOutdatedCaches(),workbox.routing.registerRoute(/\.(?:png|jpg|jpeg|gif|bmp|webp|svg|ico)$/,new workbox.strategies.CacheFirst({cacheName:"images",plugins:[new workbox.expiration.ExpirationPlugin({maxEntries:1e3,maxAgeSeconds:2592e3}),new workbox.cacheableResponse.CacheableResponsePlugin({statuses:[0,200]})]})),workbox.routing.registerRoute(/\.(?:eot|ttf|woff|woff2)$/,new workbox.strategies.CacheFirst({cacheName:"fonts",plugins:[new workbox.expiration.ExpirationPlugin({maxEntries:1e3,maxAgeSeconds:2592e3}),new workbox.cacheableResponse.CacheableResponsePlugin({statuses:[0,200]})]})),workbox.routing.registerRoute(/^https:\/\/fonts\.googleapis\.com/,new workbox.strategies.StaleWhileRevalidate({cacheName:"google-fonts-stylesheets"})),workbox.routing.registerRoute(/^https:\/\/fonts\.gstatic\.com/,new workbox.strategies.CacheFirst({cacheName:"google-fonts-webfonts",plugins:[new workbox.expiration.ExpirationPlugin({maxEntries:1e3,maxAgeSeconds:2592e3}),new workbox.cacheableResponse.CacheableResponsePlugin({statuses:[0,200]})]})),workbox.routing.registerRoute(/^https:\/\/cdn\.jsdelivr\.net/,new workbox.strategies.CacheFirst({cacheName:"static-libs",plugins:[new workbox.expiration.ExpirationPlugin({maxEntries:1e3,maxAgeSeconds:2592e3}),new workbox.cacheableResponse.CacheableResponsePlugin({statuses:[0,200]})]})),workbox.googleAnalytics.initialize();