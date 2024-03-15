[33mcommit ddb237c61cf9cc19da3627997c3691db7925a2b8[m[33m ([m[1;36mHEAD -> [m[1;32mwei/k-means[m[33m, [m[1;31morigin/wei/k-means[m[33m)[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Thu Mar 14 20:58:21 2024 -0400

    Added config switch to turn off dynamic kmeans pruning

[33mcommit 6745e1775627da4605912412e144695ae7b78906[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Thu Mar 14 20:53:50 2024 -0400

    Completed first draft of kmeans filtering

[33mcommit 41f56294e901be1fefd4f3d1ac4e262351f6d3e8[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Thu Mar 14 20:17:08 2024 -0400

    Implemented automatic switching between d-pruning and kmeans

[33mcommit 0de8271c393433810fd1bd70fe52a31343c57b22[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Thu Mar 14 19:21:00 2024 -0400

    Fixed num_agents issue in k-means prediction

[33mcommit ec300d7cc408f3b370aaa518d47034bacc8c1c8c[m
Author: Weihang Zheng <weihang.zheng@mail.utoronto.ca>
Date:   Thu Mar 14 15:46:20 2024 -0400

    added data file so you play with online_k.py

[33mcommit f9a5ffa1c7f9db75b86f87dd14daf455118e0211[m
Author: Weihang Zheng <weihang.zheng@mail.utoronto.ca>
Date:   Thu Mar 14 15:35:09 2024 -0400

    first file changes from k-means

[33mcommit 0020f1590584a41f99fbe9331f9f73e600eb1277[m
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Thu Mar 14 15:09:53 2024 -0400

    Adding capability for adversarial agents to randomize data (#55)
    
    * initial commit of a potential test setup
    
    * Added random data noise function
    
    * Removed test.py for simulation testing; irrelevant to this PR
    
    * Remove redundant print statements
    
    * Add asserts to safegaurd against logical errors
    
    * Fixed baseline agent name error
    
    ---------
    
    Co-authored-by: Weihang Zheng <weihang.zheng@mail.utoronto.ca>

[33mcommit 10659c346f51a49bd200063d1e01fb654e6ef55a[m
Author: Thanos Ariyanayagam <info@thanoshan.ca>
Date:   Tue Mar 12 22:27:08 2024 -0400

    Update RL Rewards to be local-biased & fix boundary issues (#54)
    
    * beginning experimental environment changes to prepare for RL algs - THIS WILL BREAK BASELINE ALGORITHMS FOR NOW
    
    * experimental obs implemented w/o target neighs
    
    * update requirements with raylib for experimentation
    
    * remove target neighbors from obs for now
    
    * add supersuit to requirements
    
    * add torch
    
    * temporarily remove rendering for RL experiments
    
    * more temporary experimental changes for RL
    
    * add MAPPO RL alg, NOT functional yet -- model yet to be specified. committing for cross-device access
    
    * lesson learned - sequence types dont play well with MA RL models...
    
    * add tuner
    
    * update reset to actually reset agents...
    
    * update to use ray train
    
    * bring back random agents but reset them
    
    * fix grid reset to reset properly
    
    * misc changes
    
    * dqn - untuned hyperparameters
    
    * updates to rl mappo
    
    * increase checkpointing frequenecy in DQN
    
    * update dqn
    
    * updates to algs
    
    * updates to render and experimental algs
    
    * update render
    
    * updates to NEPIADA rewards
    
    * update dqn with rainbow configuration
    
    * restore original DQN and make rainbow seperate
    
    * update requirements.txt to include CUDA torch
    
    * Transfered updation of beliefs to nepiada
    
    * update rewards to NOT use delta between states, keep scores computed and scale between 0-10, with penalties for nearing a boundary
    
    * add rewards decay linearly, can be tweaked or removed
    
    * First draft of new observations with np.arrays type
    
    * RLib trains without errors, need to add position to obs
    
    * Added positions to observations
    
    * Polished code - removed redundancy
    
    * Added config to control population of infos in baseline and some code polishing
    
    * Added torch to requirements.txt
    
    * revert agent names to adversarial and truthful since names are being passed to models anymore
    
    * remove linear decay, can be done with gamma function in rllib
    
    * update documentation in rl algs
    
    * update DQN to compress obs and implement callbacks
    
    * update epislon timesteps
    
    * print update
    
    * implement rewards change for more localized rewards and decrease gamma significantly
    
    * update iterations since last cp condition
    
    * update to config for 9x9 save
    
    * update nepiada to not clip out of bounds
    
    * fix beliefs intialization oddly reversing...
    
    * update experiment name
    
    * update legacy PPO, add target neighbours to observation space to enable parameter sharing
    
    ---------
    
    Co-authored-by: pandyah5 <pandyahetav1@gmail.com>

[33mcommit 8e0526958afed1d4eb9101d3d5dbf93b19deac49[m
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Fri Feb 16 00:14:05 2024 -0500

    Adding framework for running RL algorithms using RLib (#50)
    
    * beginning experimental environment changes to prepare for RL algs - THIS WILL BREAK BASELINE ALGORITHMS FOR NOW
    
    * experimental obs implemented w/o target neighs
    
    * update requirements with raylib for experimentation
    
    * remove target neighbors from obs for now
    
    * add supersuit to requirements
    
    * add torch
    
    * temporarily remove rendering for RL experiments
    
    * more temporary experimental changes for RL
    
    * add MAPPO RL alg, NOT functional yet -- model yet to be specified. committing for cross-device access
    
    * add mac requirements
    
    * lesson learned - sequence types dont play well with MA RL models...
    
    * add tuner
    
    * update reset to actually reset agents...
    
    * update to use ray train
    
    * bring back random agents but reset them
    
    * fix grid reset to reset properly
    
    * misc changes
    
    * dqn - untuned hyperparameters
    
    * updates to rl mappo
    
    * increase checkpointing frequenecy in DQN
    
    * update dqn
    
    * updates to algs
    
    * updates to render and experimental algs
    
    * update render
    
    * updates to NEPIADA rewards
    
    * update dqn with rainbow configuration
    
    * restore original DQN and make rainbow seperate
    
    * update requirements.txt to include CUDA torch
    
    * Transfered updation of beliefs to nepiada
    
    * update rewards to NOT use delta between states, keep scores computed and scale between 0-10, with penalties for nearing a boundary
    
    * add rewards decay linearly, can be tweaked or removed
    
    * First draft of new observations with np.arrays type
    
    * RLib trains without errors, need to add position to obs
    
    * Added positions to observations
    
    * Polished code - removed redundancy
    
    * Added config to control population of infos in baseline and some code polishing
    
    * Added torch to requirements.txt
    
    * revert agent names to adversarial and truthful since names are being passed to models anymore
    
    * remove linear decay, can be done with gamma function in rllib
    
    * update documentation in rl algs
    
    * update DQN to compress obs and implement callbacks
    
    * update epislon timesteps
    
    * print update
    
    ---------
    
    Co-authored-by: Thanoshan <info@thanoshan.ca>
    Co-authored-by: Thanoshan <36831115+Thanoshan@users.noreply.github.com>

[33mcommit 824e16e100ce3ac905ed9c5652326e210b244050[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Mon Jan 29 15:54:47 2024 -0500

    [Bypass] Update documentation in NEPIADA about dynamic communication

[33mcommit 73651bcc762ba8dc354202b8e836c6060312292f[m
Author: Thanos Ariyanayagam <info@thanoshan.ca>
Date:   Thu Jan 18 16:36:28 2024 -0500

    Implement Dynamic Communication Radius Functionality in NEPIADA (#44)
    
    * update configuration parameters to incorporate dynamic comms
    
    * update default comms radius
    
    * update config documentation
    
    * updates to algorithm and config to account for dynamic changes
    
    * rework info messages printed to console
    
    * fix naming bug in comms update
    
    * update console log statements
    
    * update documentation
    
    * minor changes to styling

[33mcommit 5123bc1a167310c95aa9be4f48e0ec6f332fb718[m
Author: Weihang Zheng <54783950+weihangzheng@users.noreply.github.com>
Date:   Tue Jan 16 08:49:22 2024 -0500

    Test Framework Setup + Convergence Score Definition (#43)
    
    * initial commit of a potential test setup
    
    * Shifted the simulation graphs under the plot folder
    
    * Modified the global rewards function to use vector distances instead of scalar
    
    * Defined convergence score as an average of agent rewards
    
    * Made convergence score non-negative, lower the better
    
    * Polished code for review
    
    * Minor edits for code quality
    
    * Addressed all comments
    
    ---------
    
    Co-authored-by: pandyah5 <pandyahetav1@gmail.com>

[33mcommit 70cc8fea1be18a23f1a70689e557bcf491a24d00[m[33m ([m[1;32mdev-hp-convergence-score[m[33m)[m
Merge: 6299758 d2083b7
Author: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
Date:   Tue Nov 28 23:54:12 2023 -0500

    Merge pull request #41 from BoundlessDevelopment/arasha/collide_grid
    
    Adding New Functionallity to Rendering

[33mcommit d2083b7a2aa10df54bbd44796405c90a45f56740[m[33m ([m[1;31morigin/arasha/collide_grid[m[33m)[m
Merge: 42ab7e6 6299758
Author: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
Date:   Tue Nov 28 23:40:33 2023 -0500

    Merge branch 'main' into arasha/collide_grid

[33mcommit 42ab7e6946a562b035cc667d147cb549678a8211[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Tue Nov 28 22:35:44 2023 -0500

    adding dynamic screen size

[33mcommit 629975839176432a67f1d6264af7ed8d089db665[m
Author: Thanos Ariyanayagam <info@thanoshan.ca>
Date:   Tue Nov 28 13:46:50 2023 -0500

    Implement a baseline algorithm (#33)
    
    * addressing comments
    
    * WIP Commit for implementing a baseline algorithm
    
    * minor changes to baseline
    
    * begin cost function implementation off agent beliefs
    
    * some documentation updates
    
    * formatting
    
    * update logic for beliefs update
    
    * fix collecting incoming messages due to wei's comment error
    
    * fix wei's comment on get all message, it's actually 'drone i is told by drone k where drone j is'
    
    * update agent step function
    
    * update documentation
    
    * update agent belief on step
    
    * add implementation for stripping D extreme values
    
    * fix a naming conflict
    
    * fix typo
    
    * fix import
    
    * resolve function signature
    
    * fix errors
    
    * fix wei's comms returning wrong dict
    
    * update logic for no estimate
    
    * fixed noise
    
    * account for invisible beliefs
    
    * allowing for collisions
    
    * addressing comments
    
    * Added weights to baseline cost function
    
    * Added an epsilon version of the baseline algorithm
    
    * Added a small marker for the target
    
    * some doc updates
    
    * fix dead imports and dead code
    
    * remove print
    
    * update episilon action selection to use env.action_space
    
    * update action space calulation in environment
    
    * address comment and fix formatting
    
    * address comments
    
    ---------
    
    Co-authored-by: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
    Co-authored-by: Weihang Zheng <weihang.zheng@mail.utoronto.ca>
    Co-authored-by: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
    Co-authored-by: pandyah5 <pandyahetav1@gmail.com>

[33mcommit 94560ef776412cf589d9bb704ff6ad6259dce138[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Mon Nov 27 17:33:49 2023 -0500

    improving rendering and adding trajectory distance dumping

[33mcommit 06750f36aa7690a126539b94d0fdcd3a7207f64c[m[33m ([m[1;31morigin/thanos/baseline[m[33m, [m[1;32mthanos/baseline[m[33m)[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Mon Nov 27 14:16:19 2023 -0500

    Added a small marker for the target

[33mcommit 2053331a9ef83eb177b0807f283ab3bf0166a6da[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Mon Nov 27 13:58:03 2023 -0500

    Added an epsilon version of the baseline algorithm

[33mcommit 8235ab9b5117b226a66b4848276b32d4a50e8409[m
Author: pandyah5 <pandyahetav1@gmail.com>
Date:   Mon Nov 27 13:50:35 2023 -0500

    Added weights to baseline cost function

[33mcommit 5eefc936fc716da0e606ee7e4e09eff355829720[m
Merge: f0deb56 9ce7409
Author: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
Date:   Sun Nov 26 21:21:32 2023 -0500

    Merge pull request #35 from BoundlessDevelopment/arasha/collide_grid
    
    removing collisions from grid

[33mcommit 9ce7409729204bac30b8546fbca24fff0d01fd3c[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Sun Nov 26 21:20:10 2023 -0500

    addressing comments

[33mcommit f6b81833dbe23cb0f6d522b65d10c44fed51115e[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Sun Nov 26 20:36:31 2023 -0500

    allowing for collisions

[33mcommit eeb43c1a5e4857e8aafc5b0c85e832f24e6eef1c[m
Merge: 1a5f931 f0deb56
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Sun Nov 26 20:29:50 2023 -0500

    Merge remote-tracking branch 'origin/thanos/baseline' into arash-render0

[33mcommit f0deb56df29e2a40a5da43c525ce55f91421f478[m
Author: Weihang Zheng <weihang.zheng@mail.utoronto.ca>
Date:   Sun Nov 26 19:38:31 2023 -0500

    account for invisible beliefs

[33mcommit c98b8c6a385bc8837a008218086488ed1ee9053f[m
Author: Weihang Zheng <weihang.zheng@mail.utoronto.ca>
Date:   Sun Nov 26 19:34:03 2023 -0500

    fixed noise

[33mcommit b6983cb5617d6eedf2d3c305a3c7195a9d6f85d8[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 16:23:31 2023 -0500

    update logic for no estimate

[33mcommit 2b54bd2690f5b7da68571c8e4a84d772b9169ad8[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 16:14:20 2023 -0500

    fix wei's comms returning wrong dict

[33mcommit 07d00301afd0ab330502556883c1018030ab69c1[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 15:44:46 2023 -0500

    fix errors

[33mcommit c65f8d6372819056ef213f8fae41b258df26facb[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 15:31:52 2023 -0500

    resolve function signature

[33mcommit 0a6a14750767dfd06c2347eb8de53d148fe3cbe2[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 15:27:27 2023 -0500

    fix import

[33mcommit d318bd07fcdae31e6d0b85b695fa4577dd44c4c4[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 15:24:44 2023 -0500

    fix typo

[33mcommit 02fb089b87871c206b3662c18544301ec7092a22[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sat Nov 25 15:22:14 2023 -0500

    fix a naming conflict

[33mcommit 91155f1e2734e15d8c2260047fa26a113e7fbd06[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Fri Nov 24 13:58:27 2023 -0500

    add implementation for stripping D extreme values

[33mcommit 54d9bd0286845be5d3940a723002bb6c6e4eec7f[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Fri Nov 24 13:48:26 2023 -0500

    update agent belief on step

[33mcommit d484dea86bb1ceb85470b6b1dd8b3e7a3beae5b0[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Fri Nov 24 13:46:55 2023 -0500

    update documentation

[33mcommit 21078378284825ab2cf8a87913d8af1c1e42b657[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:56:41 2023 -0500

    update agent step function

[33mcommit 5a2ecd708f0b88556a2b1e1f72bcd67c095a162d[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:50:31 2023 -0500

    fix wei's comment on get all message, it's actually 'drone i is told by drone k where drone j is'

[33mcommit 8f19ce645e918c9a01a85fd7a80291f4dc47a7d3[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:49:52 2023 -0500

    fix collecting incoming messages due to wei's comment error

[33mcommit b6376a17ab08c837eb548401d7270e1e5e8288a5[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:39:45 2023 -0500

    update logic for beliefs update

[33mcommit a361dfdeb3b1b77e785c20ba0aade4168c3464da[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:36:11 2023 -0500

    formatting

[33mcommit f102c7d78eaa7c0885dca54029a3e1b7b0b59a23[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:34:17 2023 -0500

    some documentation updates

[33mcommit dae1b3dbecc273cd6597306e228dcb851f7477d5[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Thu Nov 23 18:25:13 2023 -0500

    begin cost function implementation off agent beliefs

[33mcommit c5a587ce42772ffdde11865d0f8db63eeb66f805[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 23 14:45:51 2023 -0500

    minor changes to baseline

[33mcommit 77034e077fe7770eb8b7d5cfceaadf2e3cca3994[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 23 14:41:57 2023 -0500

    WIP Commit for implementing a baseline algorithm

[33mcommit 5af051e04e14850daacc7a51ee3dede23d458fb1[m
Merge: 5390fbc 75c61c8
Author: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
Date:   Wed Nov 22 21:21:48 2023 -0500

    Merge pull request #29 from BoundlessDevelopment/arash-render0
    
    Adding v0 rendering functionality

[33mcommit 1a5f931825343090285b3005be5547df5e357760[m
Merge: 9491225 75c61c8
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Wed Nov 22 21:20:46 2023 -0500

    Merge branch 'arash-render0' of https://github.com/BoundlessDevelopment/Capstone-Project into arash-render0

[33mcommit 75c61c8f510eb5274eb616f040b86cec88d78995[m[33m ([m[1;31morigin/arash-render0[m[33m)[m
Merge: f25083a 5390fbc
Author: ArashAhmadian <70601261+ArashAhmadian@users.noreply.github.com>
Date:   Wed Nov 22 21:19:59 2023 -0500

    Merge branch 'main' into arash-render0

[33mcommit 94912257514c3cd0cc9b812b2c11d162cdcb51af[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Wed Nov 22 21:17:12 2023 -0500

    addressing comments

[33mcommit 5390fbcd94be864280f1259bd629579d3f01951c[m
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Wed Nov 22 21:14:43 2023 -0500

    Implementing rewards for agents (#30)
    
    * Added target neighbour to each agent
    
    * Implemented rewards functionality
    
    * Initial setup for unit tests
    
    * Minor comments update
    
    * Implemented Thanos's comments

[33mcommit f537b4723dfb2b684f2a03ee32dacd7f9b62bd61[m
Author: Weihang Zheng <54783950+weihangzheng@users.noreply.github.com>
Date:   Mon Nov 20 17:16:22 2023 -0500

    Starting Beliefs and Communcations Implementation (#28)
    
    * starting beliefs implementation
    
    * update observation graph and belief dicts accordingly
    
    * return communicated messages as is instead of averaging them
    
    * beliefs is initialized and set as empty
    
    * move update beliefs to launch.py

[33mcommit f25083acda3daae425dd560b25a9ac6a4ef7ec62[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Tue Nov 14 20:45:07 2023 -0500

    adding animation constants

[33mcommit 34baf83b1683e23a506b83a2600076734a87430a[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Sun Nov 12 21:58:41 2023 -0500

    adding a v1 rendering implementation

[33mcommit 6cdef1df7027abfcea47cd51d4f1d1fdff534a0c[m
Author: ArashAhmadian <arash.ahmadian@mail.utoronto.ca>
Date:   Sun Nov 12 21:57:40 2023 -0500

    updating requirements

[33mcommit 1ca20bb4f21df47dce08aadc151b84e842540917[m
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Wed Nov 8 20:52:31 2023 -0500

    Implemented observation and communication graphs (#27)
    
    * Added skeleton code for obs and comm graphs
    
    * Added code to implement dynamic observation radius
    
    * Verified that observation radius is working as expected
    
    * Resolved comments on PR and added improved code flow
    
    * Error spotted - Move drone is not updating agent p_pos
    
    * Fixed error for drone not updating
    
    * Updated comments
    
    * Addressed comments about dead imports and comments

[33mcommit da332e293022679a0d0a41dc18da7e1370c37e36[m
Author: Thanos Ariyanayagam <info@thanoshan.ca>
Date:   Tue Nov 7 16:25:19 2023 -0500

    [Bypass Protection] Delete __pycache__
    
    No idea how this slipped through into the main, removing pycache files.

[33mcommit d67aaf48686128e149541358e0f5e650c4bf6e76[m
Merge: 8c33490 1c6824e
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Fri Nov 3 08:55:27 2023 -0400

    Merge pull request #26 from BoundlessDevelopment/thanos/pz_drone
    
    Implement functionality to move drones, update their positions and update the grid

[33mcommit 1c6824e724d417fc6c189aa5aa7a779446e2173e[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 2 18:45:36 2023 -0400

    update ID assignment to be global

[33mcommit c1552acd9a288846ccbc577f4ffd0a06372b388c[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 2 18:39:08 2023 -0400

    add .txt files to git ignore

[33mcommit 54c763856a9d70a3e46fc0d4adbdf77b5105d0b7[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 2 18:37:01 2023 -0400

    remove import for config in world

[33mcommit c26442f17e39a0fef717b6daa6a814fd6c56b2db[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Nov 2 18:34:30 2023 -0400

    formatting updates

[33mcommit a69c658b88ea6101f3c31c073f5e31f2a8f0f5f9[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Mon Oct 30 15:40:25 2023 -0400

    document config

[33mcommit 6c7428f4663bfe8dc24293b2daf5af54366b4137[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Oct 26 18:48:41 2023 -0400

    add assertion to verify actions

[33mcommit 22cb135b6b9dc45f2966af6ed5e0be1d2711d248[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Oct 26 17:56:17 2023 -0400

    minor nit change

[33mcommit e98902571ce8008347372e424a21f91f891eef10[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Oct 26 16:30:59 2023 -0400

    update printing

[33mcommit 9d7b41adfee5f040fefcc3d97d43fb5eac819538[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Oct 26 16:18:20 2023 -0400

    add cleaner check for if an agent is staying in the same spot

[33mcommit 37d197e40b54161e7441d6e77edcaa169df6f07a[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Thu Oct 26 16:04:03 2023 -0400

    implement first iteration of drones moving

[33mcommit e784afdce3157d53b6361c80f099763fde042069[m
Author: Thanoshan <info@thanoshan.ca>
Date:   Wed Oct 25 17:38:42 2023 -0400

    add moves dictionary to index possible moves - prep for actually moving the agents

[33mcommit 8c334903a42b412527019c768f5822acd8394f94[m
Merge: f714e66 8b6efcb
Author: Hetav Pandya <60848863+pandyah5@users.noreply.github.com>
Date:   Mon Oct 23 08:12:56 2023 -0400

    Merge pull request #20 from BoundlessDevelopment/thanos/pz_base_drone
    
    Introduce skeleton for NEPIADA environment + agents

[33mcommit 8b6efcb5c5e60861edcfd1001f0f3ffe09f51fec[m
Author: Thanoshan <36831115+Thanoshan@users.noreply.github.com>
Date:   Sun Oct 22 15:12:24 2023 -0400

    update some comments
