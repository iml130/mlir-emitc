module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 561 : i32}, tf_saved_model.semantics}  {
  func @predict(%arg0: tensor<1x224x224x3xf32> {tf._user_specified_name = "args_0", tf_saved_model.index_path = [0]}) -> (tensor<1x1000xf32> {tf_saved_model.index_path = []}) attributes {tf._input_shapes = [#tf.shape<1x224x224x3>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>, #tf.shape<*>], tf.signature.is_stateful, tf_saved_model.exported_names = ["predict"]} {
    %0 = "tosa.const"() {value = dense<0.577782452> : tensor<1000xf32>} : () -> tensor<1000xf32>
    %1 = "tosa.const"() {value = dense<0.861545979> : tensor<1280x1000xf32>} : () -> tensor<1280x1000xf32>
    %2 = "tosa.const"() {value = dense<0.283801854> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %3 = "tosa.const"() {value = dense<3.384170e-01> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %4 = "tosa.const"() {value = dense<0.480037332> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %5 = "tosa.const"() {value = dense<0.967018246> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %6 = "tosa.const"() {value = dense<0.337591648> : tensor<1x1x320x1280xf32>} : () -> tensor<1x1x320x1280xf32>
    %7 = "tosa.const"() {value = dense<0.254603893> : tensor<320xf32>} : () -> tensor<320xf32>
    %8 = "tosa.const"() {value = dense<0.321140945> : tensor<320xf32>} : () -> tensor<320xf32>
    %9 = "tosa.const"() {value = dense<0.665753961> : tensor<320xf32>} : () -> tensor<320xf32>
    %10 = "tosa.const"() {value = dense<0.843541562> : tensor<320xf32>} : () -> tensor<320xf32>
    %11 = "tosa.const"() {value = dense<6.531540e-01> : tensor<1x1x960x320xf32>} : () -> tensor<1x1x960x320xf32>
    %12 = "tosa.const"() {value = dense<0.627903759> : tensor<960xf32>} : () -> tensor<960xf32>
    %13 = "tosa.const"() {value = dense<0.644716263> : tensor<960xf32>} : () -> tensor<960xf32>
    %14 = "tosa.const"() {value = dense<0.760155618> : tensor<960xf32>} : () -> tensor<960xf32>
    %15 = "tosa.const"() {value = dense<0.871669888> : tensor<960xf32>} : () -> tensor<960xf32>
    %16 = "tosa.const"() {value = dense<0.457801312> : tensor<3x3x960x1xf32>} : () -> tensor<3x3x960x1xf32>
    %17 = "tosa.const"() {value = dense<0.796706974> : tensor<960xf32>} : () -> tensor<960xf32>
    %18 = "tosa.const"() {value = dense<0.807554125> : tensor<960xf32>} : () -> tensor<960xf32>
    %19 = "tosa.const"() {value = dense<0.88185507> : tensor<960xf32>} : () -> tensor<960xf32>
    %20 = "tosa.const"() {value = dense<0.910345196> : tensor<960xf32>} : () -> tensor<960xf32>
    %21 = "tosa.const"() {value = dense<0.564265609> : tensor<1x1x160x960xf32>} : () -> tensor<1x1x160x960xf32>
    %22 = "tosa.const"() {value = dense<0.326558739> : tensor<160xf32>} : () -> tensor<160xf32>
    %23 = "tosa.const"() {value = dense<0.796029567> : tensor<160xf32>} : () -> tensor<160xf32>
    %24 = "tosa.const"() {value = dense<0.620541573> : tensor<160xf32>} : () -> tensor<160xf32>
    %25 = "tosa.const"() {value = dense<0.664129257> : tensor<160xf32>} : () -> tensor<160xf32>
    %26 = "tosa.const"() {value = dense<0.952416598> : tensor<1x1x960x160xf32>} : () -> tensor<1x1x960x160xf32>
    %27 = "tosa.const"() {value = dense<0.60990715> : tensor<960xf32>} : () -> tensor<960xf32>
    %28 = "tosa.const"() {value = dense<0.477439612> : tensor<960xf32>} : () -> tensor<960xf32>
    %29 = "tosa.const"() {value = dense<0.288840503> : tensor<960xf32>} : () -> tensor<960xf32>
    %30 = "tosa.const"() {value = dense<0.415378749> : tensor<960xf32>} : () -> tensor<960xf32>
    %31 = "tosa.const"() {value = dense<0.199520141> : tensor<3x3x960x1xf32>} : () -> tensor<3x3x960x1xf32>
    %32 = "tosa.const"() {value = dense<0.315803587> : tensor<960xf32>} : () -> tensor<960xf32>
    %33 = "tosa.const"() {value = dense<0.269949585> : tensor<960xf32>} : () -> tensor<960xf32>
    %34 = "tosa.const"() {value = dense<0.193642944> : tensor<960xf32>} : () -> tensor<960xf32>
    %35 = "tosa.const"() {value = dense<0.471711665> : tensor<960xf32>} : () -> tensor<960xf32>
    %36 = "tosa.const"() {value = dense<0.218868583> : tensor<1x1x160x960xf32>} : () -> tensor<1x1x160x960xf32>
    %37 = "tosa.const"() {value = dense<0.1565281> : tensor<160xf32>} : () -> tensor<160xf32>
    %38 = "tosa.const"() {value = dense<0.208212137> : tensor<160xf32>} : () -> tensor<160xf32>
    %39 = "tosa.const"() {value = dense<0.292784065> : tensor<160xf32>} : () -> tensor<160xf32>
    %40 = "tosa.const"() {value = dense<0.808170139> : tensor<160xf32>} : () -> tensor<160xf32>
    %41 = "tosa.const"() {value = dense<0.268086433> : tensor<1x1x960x160xf32>} : () -> tensor<1x1x960x160xf32>
    %42 = "tosa.const"() {value = dense<0.140123621> : tensor<960xf32>} : () -> tensor<960xf32>
    %43 = "tosa.const"() {value = dense<0.355560571> : tensor<960xf32>} : () -> tensor<960xf32>
    %44 = "tosa.const"() {value = dense<0.611653864> : tensor<960xf32>} : () -> tensor<960xf32>
    %45 = "tosa.const"() {value = dense<0.318733662> : tensor<960xf32>} : () -> tensor<960xf32>
    %46 = "tosa.const"() {value = dense<0.415567845> : tensor<3x3x960x1xf32>} : () -> tensor<3x3x960x1xf32>
    %47 = "tosa.const"() {value = dense<0.364616454> : tensor<960xf32>} : () -> tensor<960xf32>
    %48 = "tosa.const"() {value = dense<0.183994696> : tensor<960xf32>} : () -> tensor<960xf32>
    %49 = "tosa.const"() {value = dense<0.353814542> : tensor<960xf32>} : () -> tensor<960xf32>
    %50 = "tosa.const"() {value = dense<0.378530592> : tensor<960xf32>} : () -> tensor<960xf32>
    %51 = "tosa.const"() {value = dense<0.725156665> : tensor<1x1x160x960xf32>} : () -> tensor<1x1x160x960xf32>
    %52 = "tosa.const"() {value = dense<0.293759793> : tensor<160xf32>} : () -> tensor<160xf32>
    %53 = "tosa.const"() {value = dense<0.614129841> : tensor<160xf32>} : () -> tensor<160xf32>
    %54 = "tosa.const"() {value = dense<0.768160343> : tensor<160xf32>} : () -> tensor<160xf32>
    %55 = "tosa.const"() {value = dense<0.978445589> : tensor<160xf32>} : () -> tensor<160xf32>
    %56 = "tosa.const"() {value = dense<0.718896747> : tensor<1x1x576x160xf32>} : () -> tensor<1x1x576x160xf32>
    %57 = "tosa.const"() {value = dense<0.728249967> : tensor<576xf32>} : () -> tensor<576xf32>
    %58 = "tosa.const"() {value = dense<0.742190182> : tensor<576xf32>} : () -> tensor<576xf32>
    %59 = "tosa.const"() {value = dense<0.957668304> : tensor<576xf32>} : () -> tensor<576xf32>
    %60 = "tosa.const"() {value = dense<0.895820856> : tensor<576xf32>} : () -> tensor<576xf32>
    %61 = "tosa.const"() {value = dense<0.673645794> : tensor<3x3x576x1xf32>} : () -> tensor<3x3x576x1xf32>
    %62 = "tosa.const"() {value = dense<0.762481689> : tensor<576xf32>} : () -> tensor<576xf32>
    %63 = "tosa.const"() {value = dense<0.290669829> : tensor<576xf32>} : () -> tensor<576xf32>
    %64 = "tosa.const"() {value = dense<0.615279614> : tensor<576xf32>} : () -> tensor<576xf32>
    %65 = "tosa.const"() {value = dense<0.496955037> : tensor<576xf32>} : () -> tensor<576xf32>
    %66 = "tosa.const"() {value = dense<9.706910e-01> : tensor<1x1x96x576xf32>} : () -> tensor<1x1x96x576xf32>
    %67 = "tosa.const"() {value = dense<0.777006804> : tensor<96xf32>} : () -> tensor<96xf32>
    %68 = "tosa.const"() {value = dense<0.903886795> : tensor<96xf32>} : () -> tensor<96xf32>
    %69 = "tosa.const"() {value = dense<0.662354589> : tensor<96xf32>} : () -> tensor<96xf32>
    %70 = "tosa.const"() {value = dense<0.463473111> : tensor<96xf32>} : () -> tensor<96xf32>
    %71 = "tosa.const"() {value = dense<0.535337508> : tensor<1x1x576x96xf32>} : () -> tensor<1x1x576x96xf32>
    %72 = "tosa.const"() {value = dense<0.285816133> : tensor<576xf32>} : () -> tensor<576xf32>
    %73 = "tosa.const"() {value = dense<0.213111147> : tensor<576xf32>} : () -> tensor<576xf32>
    %74 = "tosa.const"() {value = dense<0.468945563> : tensor<576xf32>} : () -> tensor<576xf32>
    %75 = "tosa.const"() {value = dense<0.331610262> : tensor<576xf32>} : () -> tensor<576xf32>
    %76 = "tosa.const"() {value = dense<0.941228032> : tensor<3x3x576x1xf32>} : () -> tensor<3x3x576x1xf32>
    %77 = "tosa.const"() {value = dense<0.626757204> : tensor<576xf32>} : () -> tensor<576xf32>
    %78 = "tosa.const"() {value = dense<0.750534236> : tensor<576xf32>} : () -> tensor<576xf32>
    %79 = "tosa.const"() {value = dense<0.530712128> : tensor<576xf32>} : () -> tensor<576xf32>
    %80 = "tosa.const"() {value = dense<0.733759462> : tensor<576xf32>} : () -> tensor<576xf32>
    %81 = "tosa.const"() {value = dense<0.322512716> : tensor<1x1x96x576xf32>} : () -> tensor<1x1x96x576xf32>
    %82 = "tosa.const"() {value = dense<0.465274423> : tensor<96xf32>} : () -> tensor<96xf32>
    %83 = "tosa.const"() {value = dense<0.371834457> : tensor<96xf32>} : () -> tensor<96xf32>
    %84 = "tosa.const"() {value = dense<0.927037417> : tensor<96xf32>} : () -> tensor<96xf32>
    %85 = "tosa.const"() {value = dense<0.656460702> : tensor<96xf32>} : () -> tensor<96xf32>
    %86 = "tosa.const"() {value = dense<0.19768779> : tensor<1x1x576x96xf32>} : () -> tensor<1x1x576x96xf32>
    %87 = "tosa.const"() {value = dense<0.46852085> : tensor<576xf32>} : () -> tensor<576xf32>
    %88 = "tosa.const"() {value = dense<0.113942556> : tensor<576xf32>} : () -> tensor<576xf32>
    %89 = "tosa.const"() {value = dense<0.807816684> : tensor<576xf32>} : () -> tensor<576xf32>
    %90 = "tosa.const"() {value = dense<0.388628185> : tensor<576xf32>} : () -> tensor<576xf32>
    %91 = "tosa.const"() {value = dense<0.345857173> : tensor<3x3x576x1xf32>} : () -> tensor<3x3x576x1xf32>
    %92 = "tosa.const"() {value = dense<0.0578003339> : tensor<576xf32>} : () -> tensor<576xf32>
    %93 = "tosa.const"() {value = dense<0.169078916> : tensor<576xf32>} : () -> tensor<576xf32>
    %94 = "tosa.const"() {value = dense<0.287368149> : tensor<576xf32>} : () -> tensor<576xf32>
    %95 = "tosa.const"() {value = dense<0.5602175> : tensor<576xf32>} : () -> tensor<576xf32>
    %96 = "tosa.const"() {value = dense<0.00632743444> : tensor<1x1x96x576xf32>} : () -> tensor<1x1x96x576xf32>
    %97 = "tosa.const"() {value = dense<0.476884961> : tensor<96xf32>} : () -> tensor<96xf32>
    %98 = "tosa.const"() {value = dense<0.00765841268> : tensor<96xf32>} : () -> tensor<96xf32>
    %99 = "tosa.const"() {value = dense<0.0650987476> : tensor<96xf32>} : () -> tensor<96xf32>
    %100 = "tosa.const"() {value = dense<0.695607781> : tensor<96xf32>} : () -> tensor<96xf32>
    %101 = "tosa.const"() {value = dense<0.872838616> : tensor<1x1x384x96xf32>} : () -> tensor<1x1x384x96xf32>
    %102 = "tosa.const"() {value = dense<0.13299422> : tensor<384xf32>} : () -> tensor<384xf32>
    %103 = "tosa.const"() {value = dense<0.33439362> : tensor<384xf32>} : () -> tensor<384xf32>
    %104 = "tosa.const"() {value = dense<0.156670749> : tensor<384xf32>} : () -> tensor<384xf32>
    %105 = "tosa.const"() {value = dense<0.701924622> : tensor<384xf32>} : () -> tensor<384xf32>
    %106 = "tosa.const"() {value = dense<0.442484438> : tensor<3x3x384x1xf32>} : () -> tensor<3x3x384x1xf32>
    %107 = "tosa.const"() {value = dense<0.00288255932> : tensor<384xf32>} : () -> tensor<384xf32>
    %108 = "tosa.const"() {value = dense<0.0302435737> : tensor<384xf32>} : () -> tensor<384xf32>
    %109 = "tosa.const"() {value = dense<0.333995193> : tensor<384xf32>} : () -> tensor<384xf32>
    %110 = "tosa.const"() {value = dense<0.507902741> : tensor<384xf32>} : () -> tensor<384xf32>
    %111 = "tosa.const"() {value = dense<0.835577547> : tensor<1x1x64x384xf32>} : () -> tensor<1x1x64x384xf32>
    %112 = "tosa.const"() {value = dense<0.414519638> : tensor<64xf32>} : () -> tensor<64xf32>
    %113 = "tosa.const"() {value = dense<0.950627267> : tensor<64xf32>} : () -> tensor<64xf32>
    %114 = "tosa.const"() {value = dense<0.743633687> : tensor<64xf32>} : () -> tensor<64xf32>
    %115 = "tosa.const"() {value = dense<0.357185036> : tensor<64xf32>} : () -> tensor<64xf32>
    %116 = "tosa.const"() {value = dense<0.154513195> : tensor<1x1x384x64xf32>} : () -> tensor<1x1x384x64xf32>
    %117 = "tosa.const"() {value = dense<0.781019806> : tensor<384xf32>} : () -> tensor<384xf32>
    %118 = "tosa.const"() {value = dense<0.472475469> : tensor<384xf32>} : () -> tensor<384xf32>
    %119 = "tosa.const"() {value = dense<0.764017939> : tensor<384xf32>} : () -> tensor<384xf32>
    %120 = "tosa.const"() {value = dense<0.78413403> : tensor<384xf32>} : () -> tensor<384xf32>
    %121 = "tosa.const"() {value = dense<0.299471498> : tensor<3x3x384x1xf32>} : () -> tensor<3x3x384x1xf32>
    %122 = "tosa.const"() {value = dense<0.0896641835> : tensor<384xf32>} : () -> tensor<384xf32>
    %123 = "tosa.const"() {value = dense<0.50571686> : tensor<384xf32>} : () -> tensor<384xf32>
    %124 = "tosa.const"() {value = dense<0.115420163> : tensor<384xf32>} : () -> tensor<384xf32>
    %125 = "tosa.const"() {value = dense<0.681473494> : tensor<384xf32>} : () -> tensor<384xf32>
    %126 = "tosa.const"() {value = dense<0.305642098> : tensor<1x1x64x384xf32>} : () -> tensor<1x1x64x384xf32>
    %127 = "tosa.const"() {value = dense<0.168352395> : tensor<64xf32>} : () -> tensor<64xf32>
    %128 = "tosa.const"() {value = dense<0.469091475> : tensor<64xf32>} : () -> tensor<64xf32>
    %129 = "tosa.const"() {value = dense<0.757832229> : tensor<64xf32>} : () -> tensor<64xf32>
    %130 = "tosa.const"() {value = dense<0.844817936> : tensor<64xf32>} : () -> tensor<64xf32>
    %131 = "tosa.const"() {value = dense<0.211035371> : tensor<1x1x384x64xf32>} : () -> tensor<1x1x384x64xf32>
    %132 = "tosa.const"() {value = dense<0.417507589> : tensor<384xf32>} : () -> tensor<384xf32>
    %133 = "tosa.const"() {value = dense<0.285799205> : tensor<384xf32>} : () -> tensor<384xf32>
    %134 = "tosa.const"() {value = dense<0.531456232> : tensor<384xf32>} : () -> tensor<384xf32>
    %135 = "tosa.const"() {value = dense<0.709919869> : tensor<384xf32>} : () -> tensor<384xf32>
    %136 = "tosa.const"() {value = dense<0.892971634> : tensor<3x3x384x1xf32>} : () -> tensor<3x3x384x1xf32>
    %137 = "tosa.const"() {value = dense<0.17161566> : tensor<384xf32>} : () -> tensor<384xf32>
    %138 = "tosa.const"() {value = dense<0.547968328> : tensor<384xf32>} : () -> tensor<384xf32>
    %139 = "tosa.const"() {value = dense<0.752209603> : tensor<384xf32>} : () -> tensor<384xf32>
    %140 = "tosa.const"() {value = dense<0.977758407> : tensor<384xf32>} : () -> tensor<384xf32>
    %141 = "tosa.const"() {value = dense<0.0786005631> : tensor<1x1x64x384xf32>} : () -> tensor<1x1x64x384xf32>
    %142 = "tosa.const"() {value = dense<0.048582904> : tensor<64xf32>} : () -> tensor<64xf32>
    %143 = "tosa.const"() {value = dense<0.257225901> : tensor<64xf32>} : () -> tensor<64xf32>
    %144 = "tosa.const"() {value = dense<0.873413324> : tensor<64xf32>} : () -> tensor<64xf32>
    %145 = "tosa.const"() {value = dense<0.924835741> : tensor<64xf32>} : () -> tensor<64xf32>
    %146 = "tosa.const"() {value = dense<0.78397268> : tensor<1x1x384x64xf32>} : () -> tensor<1x1x384x64xf32>
    %147 = "tosa.const"() {value = dense<0.805777609> : tensor<384xf32>} : () -> tensor<384xf32>
    %148 = "tosa.const"() {value = dense<0.219329208> : tensor<384xf32>} : () -> tensor<384xf32>
    %149 = "tosa.const"() {value = dense<0.352058202> : tensor<384xf32>} : () -> tensor<384xf32>
    %150 = "tosa.const"() {value = dense<0.157239452> : tensor<384xf32>} : () -> tensor<384xf32>
    %151 = "tosa.const"() {value = dense<0.489881128> : tensor<3x3x384x1xf32>} : () -> tensor<3x3x384x1xf32>
    %152 = "tosa.const"() {value = dense<0.685616672> : tensor<384xf32>} : () -> tensor<384xf32>
    %153 = "tosa.const"() {value = dense<0.336946934> : tensor<384xf32>} : () -> tensor<384xf32>
    %154 = "tosa.const"() {value = dense<0.0249882974> : tensor<384xf32>} : () -> tensor<384xf32>
    %155 = "tosa.const"() {value = dense<0.96508193> : tensor<384xf32>} : () -> tensor<384xf32>
    %156 = "tosa.const"() {value = dense<0.729144573> : tensor<1x1x64x384xf32>} : () -> tensor<1x1x64x384xf32>
    %157 = "tosa.const"() {value = dense<0.0154308025> : tensor<64xf32>} : () -> tensor<64xf32>
    %158 = "tosa.const"() {value = dense<0.275488228> : tensor<64xf32>} : () -> tensor<64xf32>
    %159 = "tosa.const"() {value = dense<0.958814561> : tensor<64xf32>} : () -> tensor<64xf32>
    %160 = "tosa.const"() {value = dense<0.0950143262> : tensor<64xf32>} : () -> tensor<64xf32>
    %161 = "tosa.const"() {value = dense<0.239749134> : tensor<1x1x192x64xf32>} : () -> tensor<1x1x192x64xf32>
    %162 = "tosa.const"() {value = dense<0.291229665> : tensor<192xf32>} : () -> tensor<192xf32>
    %163 = "tosa.const"() {value = dense<0.859531044> : tensor<192xf32>} : () -> tensor<192xf32>
    %164 = "tosa.const"() {value = dense<0.312232822> : tensor<192xf32>} : () -> tensor<192xf32>
    %165 = "tosa.const"() {value = dense<0.776254117> : tensor<192xf32>} : () -> tensor<192xf32>
    %166 = "tosa.const"() {value = dense<0.590008795> : tensor<3x3x192x1xf32>} : () -> tensor<3x3x192x1xf32>
    %167 = "tosa.const"() {value = dense<0.818245887> : tensor<192xf32>} : () -> tensor<192xf32>
    %168 = "tosa.const"() {value = dense<0.925683856> : tensor<192xf32>} : () -> tensor<192xf32>
    %169 = "tosa.const"() {value = dense<0.447279423> : tensor<192xf32>} : () -> tensor<192xf32>
    %170 = "tosa.const"() {value = dense<8.773110e-01> : tensor<192xf32>} : () -> tensor<192xf32>
    %171 = "tosa.const"() {value = dense<0.761288166> : tensor<1x1x32x192xf32>} : () -> tensor<1x1x32x192xf32>
    %172 = "tosa.const"() {value = dense<0.291238338> : tensor<32xf32>} : () -> tensor<32xf32>
    %173 = "tosa.const"() {value = dense<0.911299407> : tensor<32xf32>} : () -> tensor<32xf32>
    %174 = "tosa.const"() {value = dense<7.569580e-01> : tensor<32xf32>} : () -> tensor<32xf32>
    %175 = "tosa.const"() {value = dense<0.611750603> : tensor<32xf32>} : () -> tensor<32xf32>
    %176 = "tosa.const"() {value = dense<0.633168459> : tensor<1x1x192x32xf32>} : () -> tensor<1x1x192x32xf32>
    %177 = "tosa.const"() {value = dense<0.746092558> : tensor<192xf32>} : () -> tensor<192xf32>
    %178 = "tosa.const"() {value = dense<0.0902657508> : tensor<192xf32>} : () -> tensor<192xf32>
    %179 = "tosa.const"() {value = dense<0.787089109> : tensor<192xf32>} : () -> tensor<192xf32>
    %180 = "tosa.const"() {value = dense<0.967407107> : tensor<192xf32>} : () -> tensor<192xf32>
    %181 = "tosa.const"() {value = dense<0.272493422> : tensor<3x3x192x1xf32>} : () -> tensor<3x3x192x1xf32>
    %182 = "tosa.const"() {value = dense<0.342552841> : tensor<192xf32>} : () -> tensor<192xf32>
    %183 = "tosa.const"() {value = dense<0.258820027> : tensor<192xf32>} : () -> tensor<192xf32>
    %184 = "tosa.const"() {value = dense<0.202512547> : tensor<192xf32>} : () -> tensor<192xf32>
    %185 = "tosa.const"() {value = dense<0.930393517> : tensor<192xf32>} : () -> tensor<192xf32>
    %186 = "tosa.const"() {value = dense<0.291055024> : tensor<1x1x32x192xf32>} : () -> tensor<1x1x32x192xf32>
    %187 = "tosa.const"() {value = dense<0.869512259> : tensor<32xf32>} : () -> tensor<32xf32>
    %188 = "tosa.const"() {value = dense<0.312103331> : tensor<32xf32>} : () -> tensor<32xf32>
    %189 = "tosa.const"() {value = dense<0.308277726> : tensor<32xf32>} : () -> tensor<32xf32>
    %190 = "tosa.const"() {value = dense<0.741228401> : tensor<32xf32>} : () -> tensor<32xf32>
    %191 = "tosa.const"() {value = dense<0.842454135> : tensor<1x1x192x32xf32>} : () -> tensor<1x1x192x32xf32>
    %192 = "tosa.const"() {value = dense<0.760335445> : tensor<192xf32>} : () -> tensor<192xf32>
    %193 = "tosa.const"() {value = dense<0.372991323> : tensor<192xf32>} : () -> tensor<192xf32>
    %194 = "tosa.const"() {value = dense<0.446741521> : tensor<192xf32>} : () -> tensor<192xf32>
    %195 = "tosa.const"() {value = dense<0.165641442> : tensor<192xf32>} : () -> tensor<192xf32>
    %196 = "tosa.const"() {value = dense<0.507602274> : tensor<3x3x192x1xf32>} : () -> tensor<3x3x192x1xf32>
    %197 = "tosa.const"() {value = dense<5.050440e-01> : tensor<192xf32>} : () -> tensor<192xf32>
    %198 = "tosa.const"() {value = dense<0.962219417> : tensor<192xf32>} : () -> tensor<192xf32>
    %199 = "tosa.const"() {value = dense<0.808169186> : tensor<192xf32>} : () -> tensor<192xf32>
    %200 = "tosa.const"() {value = dense<0.0506269373> : tensor<192xf32>} : () -> tensor<192xf32>
    %201 = "tosa.const"() {value = dense<0.539532125> : tensor<1x1x32x192xf32>} : () -> tensor<1x1x32x192xf32>
    %202 = "tosa.const"() {value = dense<0.484412521> : tensor<32xf32>} : () -> tensor<32xf32>
    %203 = "tosa.const"() {value = dense<0.98500961> : tensor<32xf32>} : () -> tensor<32xf32>
    %204 = "tosa.const"() {value = dense<0.571860194> : tensor<32xf32>} : () -> tensor<32xf32>
    %205 = "tosa.const"() {value = dense<0.0859645307> : tensor<32xf32>} : () -> tensor<32xf32>
    %206 = "tosa.const"() {value = dense<0.477637827> : tensor<1x1x144x32xf32>} : () -> tensor<1x1x144x32xf32>
    %207 = "tosa.const"() {value = dense<0.732360541> : tensor<144xf32>} : () -> tensor<144xf32>
    %208 = "tosa.const"() {value = dense<0.738098204> : tensor<144xf32>} : () -> tensor<144xf32>
    %209 = "tosa.const"() {value = dense<0.24417983> : tensor<144xf32>} : () -> tensor<144xf32>
    %210 = "tosa.const"() {value = dense<0.628451109> : tensor<144xf32>} : () -> tensor<144xf32>
    %211 = "tosa.const"() {value = dense<0.711884438> : tensor<3x3x144x1xf32>} : () -> tensor<3x3x144x1xf32>
    %212 = "tosa.const"() {value = dense<0.314113259> : tensor<144xf32>} : () -> tensor<144xf32>
    %213 = "tosa.const"() {value = dense<0.0668376908> : tensor<144xf32>} : () -> tensor<144xf32>
    %214 = "tosa.const"() {value = dense<0.919994533> : tensor<144xf32>} : () -> tensor<144xf32>
    %215 = "tosa.const"() {value = dense<0.476946533> : tensor<144xf32>} : () -> tensor<144xf32>
    %216 = "tosa.const"() {value = dense<0.842674195> : tensor<1x1x24x144xf32>} : () -> tensor<1x1x24x144xf32>
    %217 = "tosa.const"() {value = dense<0.27963388> : tensor<24xf32>} : () -> tensor<24xf32>
    %218 = "tosa.const"() {value = dense<0.119340651> : tensor<24xf32>} : () -> tensor<24xf32>
    %219 = "tosa.const"() {value = dense<0.292897433> : tensor<24xf32>} : () -> tensor<24xf32>
    %220 = "tosa.const"() {value = dense<0.357308686> : tensor<24xf32>} : () -> tensor<24xf32>
    %221 = "tosa.const"() {value = dense<0.653880298> : tensor<1x1x144x24xf32>} : () -> tensor<1x1x144x24xf32>
    %222 = "tosa.const"() {value = dense<0.129047826> : tensor<144xf32>} : () -> tensor<144xf32>
    %223 = "tosa.const"() {value = dense<0.125544801> : tensor<144xf32>} : () -> tensor<144xf32>
    %224 = "tosa.const"() {value = dense<0.635462462> : tensor<144xf32>} : () -> tensor<144xf32>
    %225 = "tosa.const"() {value = dense<0.689478397> : tensor<144xf32>} : () -> tensor<144xf32>
    %226 = "tosa.const"() {value = dense<0.830536723> : tensor<3x3x144x1xf32>} : () -> tensor<3x3x144x1xf32>
    %227 = "tosa.const"() {value = dense<0.136199415> : tensor<144xf32>} : () -> tensor<144xf32>
    %228 = "tosa.const"() {value = dense<0.749371826> : tensor<144xf32>} : () -> tensor<144xf32>
    %229 = "tosa.const"() {value = dense<0.912112891> : tensor<144xf32>} : () -> tensor<144xf32>
    %230 = "tosa.const"() {value = dense<0.611115276> : tensor<144xf32>} : () -> tensor<144xf32>
    %231 = "tosa.const"() {value = dense<0.400357842> : tensor<1x1x24x144xf32>} : () -> tensor<1x1x24x144xf32>
    %232 = "tosa.const"() {value = dense<0.505576909> : tensor<24xf32>} : () -> tensor<24xf32>
    %233 = "tosa.const"() {value = dense<0.501250207> : tensor<24xf32>} : () -> tensor<24xf32>
    %234 = "tosa.const"() {value = dense<0.928310573> : tensor<24xf32>} : () -> tensor<24xf32>
    %235 = "tosa.const"() {value = dense<0.141835421> : tensor<24xf32>} : () -> tensor<24xf32>
    %236 = "tosa.const"() {value = dense<0.445325881> : tensor<1x1x96x24xf32>} : () -> tensor<1x1x96x24xf32>
    %237 = "tosa.const"() {value = dense<0.411665559> : tensor<96xf32>} : () -> tensor<96xf32>
    %238 = "tosa.const"() {value = dense<0.0536422394> : tensor<96xf32>} : () -> tensor<96xf32>
    %239 = "tosa.const"() {value = dense<0.17796351> : tensor<96xf32>} : () -> tensor<96xf32>
    %240 = "tosa.const"() {value = dense<0.309271187> : tensor<96xf32>} : () -> tensor<96xf32>
    %241 = "tosa.const"() {value = dense<0.00657993509> : tensor<3x3x96x1xf32>} : () -> tensor<3x3x96x1xf32>
    %242 = "tosa.const"() {value = dense<0.473965913> : tensor<96xf32>} : () -> tensor<96xf32>
    %243 = "tosa.const"() {value = dense<0.693110764> : tensor<96xf32>} : () -> tensor<96xf32>
    %244 = "tosa.const"() {value = dense<0.627719879> : tensor<96xf32>} : () -> tensor<96xf32>
    %245 = "tosa.const"() {value = dense<0.749659478> : tensor<96xf32>} : () -> tensor<96xf32>
    %246 = "tosa.const"() {value = dense<0.216990724> : tensor<1x1x16x96xf32>} : () -> tensor<1x1x16x96xf32>
    %247 = "tosa.const"() {value = dense<0.345583141> : tensor<16xf32>} : () -> tensor<16xf32>
    %248 = "tosa.const"() {value = dense<0.545788586> : tensor<16xf32>} : () -> tensor<16xf32>
    %249 = "tosa.const"() {value = dense<0.954604923> : tensor<16xf32>} : () -> tensor<16xf32>
    %250 = "tosa.const"() {value = dense<0.694585144> : tensor<16xf32>} : () -> tensor<16xf32>
    %251 = "tosa.const"() {value = dense<0.350052834> : tensor<1x1x32x16xf32>} : () -> tensor<1x1x32x16xf32>
    %252 = "tosa.const"() {value = dense<0.511850059> : tensor<32xf32>} : () -> tensor<32xf32>
    %253 = "tosa.const"() {value = dense<0.944634616> : tensor<32xf32>} : () -> tensor<32xf32>
    %254 = "tosa.const"() {value = dense<0.471779138> : tensor<32xf32>} : () -> tensor<32xf32>
    %255 = "tosa.const"() {value = dense<0.229005009> : tensor<32xf32>} : () -> tensor<32xf32>
    %256 = "tosa.const"() {value = dense<0.12572439> : tensor<3x3x32x1xf32>} : () -> tensor<3x3x32x1xf32>
    %257 = "tosa.const"() {value = dense<0.420583606> : tensor<32xf32>} : () -> tensor<32xf32>
    %258 = "tosa.const"() {value = dense<0.269001395> : tensor<32xf32>} : () -> tensor<32xf32>
    %259 = "tosa.const"() {value = dense<0.798075199> : tensor<32xf32>} : () -> tensor<32xf32>
    %260 = "tosa.const"() {value = dense<0.791799962> : tensor<32xf32>} : () -> tensor<32xf32>
    %261 = "tosa.const"() {value = dense<0.795556604> : tensor<3x3x3x32xf32>} : () -> tensor<3x3x3x32xf32>
    %262 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %263 = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %264 = "tosa.const"() {value = dense<0.000000e+00> : tensor<24xf32>} : () -> tensor<24xf32>
    %265 = "tosa.const"() {value = dense<0.000000e+00> : tensor<144xf32>} : () -> tensor<144xf32>
    %266 = "tosa.const"() {value = dense<0.000000e+00> : tensor<32xf32>} : () -> tensor<32xf32>
    %267 = "tosa.const"() {value = dense<0.000000e+00> : tensor<192xf32>} : () -> tensor<192xf32>
    %268 = "tosa.const"() {value = dense<0.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
    %269 = "tosa.const"() {value = dense<0.000000e+00> : tensor<384xf32>} : () -> tensor<384xf32>
    %270 = "tosa.const"() {value = dense<0.000000e+00> : tensor<96xf32>} : () -> tensor<96xf32>
    %271 = "tosa.const"() {value = dense<0.000000e+00> : tensor<576xf32>} : () -> tensor<576xf32>
    %272 = "tosa.const"() {value = dense<0.000000e+00> : tensor<160xf32>} : () -> tensor<160xf32>
    %273 = "tosa.const"() {value = dense<0.000000e+00> : tensor<960xf32>} : () -> tensor<960xf32>
    %274 = "tosa.const"() {value = dense<0.000000e+00> : tensor<320xf32>} : () -> tensor<320xf32>
    %275 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %276 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %277 = "tosa.const"() {value = dense<1.000000e-03> : tensor<1xf32>} : () -> tensor<1xf32>
    %278 = "tosa.const"() {value = dense<0.0204081628> : tensor<f32>} : () -> tensor<f32>
    %279 = "tosa.transpose"(%261, %276) : (tensor<3x3x3x32xf32>, tensor<4xi32>) -> tensor<32x3x3x3xf32>
    %280 = "tosa.conv2d"(%arg0, %279, %266) {dilation = [1, 1], pad = [0, 1, 0, 1], stride = [2, 2]} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %281 = "tosa.reshape"(%258) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %282 = "tosa.sub"(%280, %281) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %283 = "tosa.add"(%257, %277) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %284 = "tosa.rsqrt"(%283) : (tensor<32xf32>) -> tensor<32xf32>
    %285 = "tosa.reshape"(%284) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %286 = "tosa.mul"(%282, %285) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %287 = "tosa.reshape"(%260) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %288 = "tosa.mul"(%286, %287) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %289 = "tosa.reshape"(%259) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %290 = "tosa.add"(%288, %289) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %291 = "tosa.reluN"(%290) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %292 = "tosa.depthwise_conv2d"(%291, %256, %266) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %293 = "tosa.reshape"(%253) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %294 = "tosa.sub"(%292, %293) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %295 = "tosa.add"(%252, %277) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %296 = "tosa.rsqrt"(%295) : (tensor<32xf32>) -> tensor<32xf32>
    %297 = "tosa.reshape"(%296) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %298 = "tosa.mul"(%294, %297) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %299 = "tosa.reshape"(%255) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %300 = "tosa.mul"(%298, %299) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %301 = "tosa.reshape"(%254) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %302 = "tosa.add"(%300, %301) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %303 = "tosa.reluN"(%302) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %304 = "tosa.transpose"(%251, %276) : (tensor<1x1x32x16xf32>, tensor<4xi32>) -> tensor<16x1x1x32xf32>
    %305 = "tosa.conv2d"(%303, %304, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x112x112x32xf32>, tensor<16x1x1x32xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %306 = "tosa.reshape"(%248) {new_shape = [1, 1, 1, 16]} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %307 = "tosa.sub"(%305, %306) : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %308 = "tosa.add"(%247, %277) : (tensor<16xf32>, tensor<1xf32>) -> tensor<16xf32>
    %309 = "tosa.rsqrt"(%308) : (tensor<16xf32>) -> tensor<16xf32>
    %310 = "tosa.reshape"(%309) {new_shape = [1, 1, 1, 16]} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %311 = "tosa.mul"(%307, %310) {shift = 0 : i32} : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %312 = "tosa.reshape"(%250) {new_shape = [1, 1, 1, 16]} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %313 = "tosa.mul"(%311, %312) {shift = 0 : i32} : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %314 = "tosa.reshape"(%249) {new_shape = [1, 1, 1, 16]} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %315 = "tosa.add"(%313, %314) : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %316 = "tosa.transpose"(%246, %276) : (tensor<1x1x16x96xf32>, tensor<4xi32>) -> tensor<96x1x1x16xf32>
    %317 = "tosa.conv2d"(%315, %316, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x112x112x16xf32>, tensor<96x1x1x16xf32>, tensor<96xf32>) -> tensor<1x112x112x96xf32>
    %318 = "tosa.reshape"(%243) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %319 = "tosa.sub"(%317, %318) : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %320 = "tosa.add"(%242, %277) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %321 = "tosa.rsqrt"(%320) : (tensor<96xf32>) -> tensor<96xf32>
    %322 = "tosa.reshape"(%321) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %323 = "tosa.mul"(%319, %322) {shift = 0 : i32} : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %324 = "tosa.reshape"(%245) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %325 = "tosa.mul"(%323, %324) {shift = 0 : i32} : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %326 = "tosa.reshape"(%244) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %327 = "tosa.add"(%325, %326) : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %328 = "tosa.reluN"(%327) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
    %329 = "tosa.pad"(%328, %262) : (tensor<1x112x112x96xf32>, tensor<4x2xi32>) -> tensor<1x113x113x96xf32>
    %330 = "tosa.depthwise_conv2d"(%329, %241, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x113x113x96xf32>, tensor<3x3x96x1xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %331 = "tosa.reshape"(%238) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %332 = "tosa.sub"(%330, %331) : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %333 = "tosa.add"(%237, %277) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %334 = "tosa.rsqrt"(%333) : (tensor<96xf32>) -> tensor<96xf32>
    %335 = "tosa.reshape"(%334) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %336 = "tosa.mul"(%332, %335) {shift = 0 : i32} : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %337 = "tosa.reshape"(%240) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %338 = "tosa.mul"(%336, %337) {shift = 0 : i32} : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %339 = "tosa.reshape"(%239) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %340 = "tosa.add"(%338, %339) : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %341 = "tosa.reluN"(%340) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    %342 = "tosa.transpose"(%236, %276) : (tensor<1x1x96x24xf32>, tensor<4xi32>) -> tensor<24x1x1x96xf32>
    %343 = "tosa.conv2d"(%341, %342, %264) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x56x56x96xf32>, tensor<24x1x1x96xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %344 = "tosa.reshape"(%233) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %345 = "tosa.sub"(%343, %344) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %346 = "tosa.add"(%232, %277) : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
    %347 = "tosa.rsqrt"(%346) : (tensor<24xf32>) -> tensor<24xf32>
    %348 = "tosa.reshape"(%347) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %349 = "tosa.mul"(%345, %348) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %350 = "tosa.reshape"(%235) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %351 = "tosa.mul"(%349, %350) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %352 = "tosa.reshape"(%234) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %353 = "tosa.add"(%351, %352) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %354 = "tosa.transpose"(%231, %276) : (tensor<1x1x24x144xf32>, tensor<4xi32>) -> tensor<144x1x1x24xf32>
    %355 = "tosa.conv2d"(%353, %354, %265) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x56x56x24xf32>, tensor<144x1x1x24xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %356 = "tosa.reshape"(%228) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %357 = "tosa.sub"(%355, %356) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %358 = "tosa.add"(%227, %277) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %359 = "tosa.rsqrt"(%358) : (tensor<144xf32>) -> tensor<144xf32>
    %360 = "tosa.reshape"(%359) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %361 = "tosa.mul"(%357, %360) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %362 = "tosa.reshape"(%230) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %363 = "tosa.mul"(%361, %362) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %364 = "tosa.reshape"(%229) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %365 = "tosa.add"(%363, %364) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %366 = "tosa.reluN"(%365) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %367 = "tosa.depthwise_conv2d"(%366, %226, %265) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x56x56x144xf32>, tensor<3x3x144x1xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %368 = "tosa.reshape"(%223) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %369 = "tosa.sub"(%367, %368) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %370 = "tosa.add"(%222, %277) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %371 = "tosa.rsqrt"(%370) : (tensor<144xf32>) -> tensor<144xf32>
    %372 = "tosa.reshape"(%371) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %373 = "tosa.mul"(%369, %372) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %374 = "tosa.reshape"(%225) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %375 = "tosa.mul"(%373, %374) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %376 = "tosa.reshape"(%224) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %377 = "tosa.add"(%375, %376) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %378 = "tosa.reluN"(%377) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %379 = "tosa.transpose"(%221, %276) : (tensor<1x1x144x24xf32>, tensor<4xi32>) -> tensor<24x1x1x144xf32>
    %380 = "tosa.conv2d"(%378, %379, %264) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x56x56x144xf32>, tensor<24x1x1x144xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %381 = "tosa.reshape"(%218) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %382 = "tosa.sub"(%380, %381) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %383 = "tosa.add"(%217, %277) : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
    %384 = "tosa.rsqrt"(%383) : (tensor<24xf32>) -> tensor<24xf32>
    %385 = "tosa.reshape"(%384) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %386 = "tosa.mul"(%382, %385) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %387 = "tosa.reshape"(%220) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %388 = "tosa.mul"(%386, %387) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %389 = "tosa.reshape"(%219) {new_shape = [1, 1, 1, 24]} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %390 = "tosa.add"(%388, %389) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %391 = "tosa.add"(%353, %390) : (tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
    %392 = "tosa.transpose"(%216, %276) : (tensor<1x1x24x144xf32>, tensor<4xi32>) -> tensor<144x1x1x24xf32>
    %393 = "tosa.conv2d"(%391, %392, %265) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x56x56x24xf32>, tensor<144x1x1x24xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %394 = "tosa.reshape"(%213) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %395 = "tosa.sub"(%393, %394) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %396 = "tosa.add"(%212, %277) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %397 = "tosa.rsqrt"(%396) : (tensor<144xf32>) -> tensor<144xf32>
    %398 = "tosa.reshape"(%397) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %399 = "tosa.mul"(%395, %398) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %400 = "tosa.reshape"(%215) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %401 = "tosa.mul"(%399, %400) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %402 = "tosa.reshape"(%214) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %403 = "tosa.add"(%401, %402) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %404 = "tosa.reluN"(%403) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %405 = "tosa.pad"(%404, %262) : (tensor<1x56x56x144xf32>, tensor<4x2xi32>) -> tensor<1x57x57x144xf32>
    %406 = "tosa.depthwise_conv2d"(%405, %211, %265) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x57x57x144xf32>, tensor<3x3x144x1xf32>, tensor<144xf32>) -> tensor<1x28x28x144xf32>
    %407 = "tosa.reshape"(%208) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %408 = "tosa.sub"(%406, %407) : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %409 = "tosa.add"(%207, %277) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %410 = "tosa.rsqrt"(%409) : (tensor<144xf32>) -> tensor<144xf32>
    %411 = "tosa.reshape"(%410) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %412 = "tosa.mul"(%408, %411) {shift = 0 : i32} : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %413 = "tosa.reshape"(%210) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %414 = "tosa.mul"(%412, %413) {shift = 0 : i32} : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %415 = "tosa.reshape"(%209) {new_shape = [1, 1, 1, 144]} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %416 = "tosa.add"(%414, %415) : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %417 = "tosa.reluN"(%416) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
    %418 = "tosa.transpose"(%206, %276) : (tensor<1x1x144x32xf32>, tensor<4xi32>) -> tensor<32x1x1x144xf32>
    %419 = "tosa.conv2d"(%417, %418, %266) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x144xf32>, tensor<32x1x1x144xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %420 = "tosa.reshape"(%203) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %421 = "tosa.sub"(%419, %420) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %422 = "tosa.add"(%202, %277) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %423 = "tosa.rsqrt"(%422) : (tensor<32xf32>) -> tensor<32xf32>
    %424 = "tosa.reshape"(%423) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %425 = "tosa.mul"(%421, %424) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %426 = "tosa.reshape"(%205) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %427 = "tosa.mul"(%425, %426) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %428 = "tosa.reshape"(%204) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %429 = "tosa.add"(%427, %428) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %430 = "tosa.transpose"(%201, %276) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %431 = "tosa.conv2d"(%429, %430, %267) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %432 = "tosa.reshape"(%198) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %433 = "tosa.sub"(%431, %432) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %434 = "tosa.add"(%197, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %435 = "tosa.rsqrt"(%434) : (tensor<192xf32>) -> tensor<192xf32>
    %436 = "tosa.reshape"(%435) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %437 = "tosa.mul"(%433, %436) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %438 = "tosa.reshape"(%200) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %439 = "tosa.mul"(%437, %438) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %440 = "tosa.reshape"(%199) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %441 = "tosa.add"(%439, %440) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %442 = "tosa.reluN"(%441) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %443 = "tosa.depthwise_conv2d"(%442, %196, %267) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x28x28x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %444 = "tosa.reshape"(%193) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %445 = "tosa.sub"(%443, %444) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %446 = "tosa.add"(%192, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %447 = "tosa.rsqrt"(%446) : (tensor<192xf32>) -> tensor<192xf32>
    %448 = "tosa.reshape"(%447) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %449 = "tosa.mul"(%445, %448) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %450 = "tosa.reshape"(%195) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %451 = "tosa.mul"(%449, %450) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %452 = "tosa.reshape"(%194) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %453 = "tosa.add"(%451, %452) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %454 = "tosa.reluN"(%453) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %455 = "tosa.transpose"(%191, %276) : (tensor<1x1x192x32xf32>, tensor<4xi32>) -> tensor<32x1x1x192xf32>
    %456 = "tosa.conv2d"(%454, %455, %266) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x192xf32>, tensor<32x1x1x192xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %457 = "tosa.reshape"(%188) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %458 = "tosa.sub"(%456, %457) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %459 = "tosa.add"(%187, %277) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %460 = "tosa.rsqrt"(%459) : (tensor<32xf32>) -> tensor<32xf32>
    %461 = "tosa.reshape"(%460) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %462 = "tosa.mul"(%458, %461) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %463 = "tosa.reshape"(%190) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %464 = "tosa.mul"(%462, %463) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %465 = "tosa.reshape"(%189) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %466 = "tosa.add"(%464, %465) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %467 = "tosa.add"(%429, %466) : (tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
    %468 = "tosa.transpose"(%186, %276) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %469 = "tosa.conv2d"(%467, %468, %267) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %470 = "tosa.reshape"(%183) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %471 = "tosa.sub"(%469, %470) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %472 = "tosa.add"(%182, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %473 = "tosa.rsqrt"(%472) : (tensor<192xf32>) -> tensor<192xf32>
    %474 = "tosa.reshape"(%473) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %475 = "tosa.mul"(%471, %474) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %476 = "tosa.reshape"(%185) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %477 = "tosa.mul"(%475, %476) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %478 = "tosa.reshape"(%184) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %479 = "tosa.add"(%477, %478) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %480 = "tosa.reluN"(%479) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %481 = "tosa.depthwise_conv2d"(%480, %181, %267) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x28x28x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %482 = "tosa.reshape"(%178) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %483 = "tosa.sub"(%481, %482) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %484 = "tosa.add"(%177, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %485 = "tosa.rsqrt"(%484) : (tensor<192xf32>) -> tensor<192xf32>
    %486 = "tosa.reshape"(%485) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %487 = "tosa.mul"(%483, %486) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %488 = "tosa.reshape"(%180) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %489 = "tosa.mul"(%487, %488) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %490 = "tosa.reshape"(%179) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %491 = "tosa.add"(%489, %490) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %492 = "tosa.reluN"(%491) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %493 = "tosa.transpose"(%176, %276) : (tensor<1x1x192x32xf32>, tensor<4xi32>) -> tensor<32x1x1x192xf32>
    %494 = "tosa.conv2d"(%492, %493, %266) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x192xf32>, tensor<32x1x1x192xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %495 = "tosa.reshape"(%173) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %496 = "tosa.sub"(%494, %495) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %497 = "tosa.add"(%172, %277) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %498 = "tosa.rsqrt"(%497) : (tensor<32xf32>) -> tensor<32xf32>
    %499 = "tosa.reshape"(%498) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %500 = "tosa.mul"(%496, %499) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %501 = "tosa.reshape"(%175) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %502 = "tosa.mul"(%500, %501) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %503 = "tosa.reshape"(%174) {new_shape = [1, 1, 1, 32]} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %504 = "tosa.add"(%502, %503) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %505 = "tosa.add"(%467, %504) : (tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
    %506 = "tosa.transpose"(%171, %276) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %507 = "tosa.conv2d"(%505, %506, %267) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %508 = "tosa.reshape"(%168) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %509 = "tosa.sub"(%507, %508) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %510 = "tosa.add"(%167, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %511 = "tosa.rsqrt"(%510) : (tensor<192xf32>) -> tensor<192xf32>
    %512 = "tosa.reshape"(%511) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %513 = "tosa.mul"(%509, %512) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %514 = "tosa.reshape"(%170) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %515 = "tosa.mul"(%513, %514) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %516 = "tosa.reshape"(%169) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %517 = "tosa.add"(%515, %516) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %518 = "tosa.reluN"(%517) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %519 = "tosa.pad"(%518, %262) : (tensor<1x28x28x192xf32>, tensor<4x2xi32>) -> tensor<1x29x29x192xf32>
    %520 = "tosa.depthwise_conv2d"(%519, %166, %267) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x29x29x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x14x14x192xf32>
    %521 = "tosa.reshape"(%163) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %522 = "tosa.sub"(%520, %521) : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %523 = "tosa.add"(%162, %277) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %524 = "tosa.rsqrt"(%523) : (tensor<192xf32>) -> tensor<192xf32>
    %525 = "tosa.reshape"(%524) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %526 = "tosa.mul"(%522, %525) {shift = 0 : i32} : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %527 = "tosa.reshape"(%165) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %528 = "tosa.mul"(%526, %527) {shift = 0 : i32} : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %529 = "tosa.reshape"(%164) {new_shape = [1, 1, 1, 192]} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %530 = "tosa.add"(%528, %529) : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %531 = "tosa.reluN"(%530) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
    %532 = "tosa.transpose"(%161, %276) : (tensor<1x1x192x64xf32>, tensor<4xi32>) -> tensor<64x1x1x192xf32>
    %533 = "tosa.conv2d"(%531, %532, %268) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x192xf32>, tensor<64x1x1x192xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %534 = "tosa.reshape"(%158) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %535 = "tosa.sub"(%533, %534) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %536 = "tosa.add"(%157, %277) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %537 = "tosa.rsqrt"(%536) : (tensor<64xf32>) -> tensor<64xf32>
    %538 = "tosa.reshape"(%537) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %539 = "tosa.mul"(%535, %538) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %540 = "tosa.reshape"(%160) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %541 = "tosa.mul"(%539, %540) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %542 = "tosa.reshape"(%159) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %543 = "tosa.add"(%541, %542) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %544 = "tosa.transpose"(%156, %276) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %545 = "tosa.conv2d"(%543, %544, %269) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %546 = "tosa.reshape"(%153) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %547 = "tosa.sub"(%545, %546) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %548 = "tosa.add"(%152, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %549 = "tosa.rsqrt"(%548) : (tensor<384xf32>) -> tensor<384xf32>
    %550 = "tosa.reshape"(%549) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %551 = "tosa.mul"(%547, %550) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %552 = "tosa.reshape"(%155) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %553 = "tosa.mul"(%551, %552) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %554 = "tosa.reshape"(%154) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %555 = "tosa.add"(%553, %554) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %556 = "tosa.reluN"(%555) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %557 = "tosa.depthwise_conv2d"(%556, %151, %269) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %558 = "tosa.reshape"(%148) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %559 = "tosa.sub"(%557, %558) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %560 = "tosa.add"(%147, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %561 = "tosa.rsqrt"(%560) : (tensor<384xf32>) -> tensor<384xf32>
    %562 = "tosa.reshape"(%561) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %563 = "tosa.mul"(%559, %562) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %564 = "tosa.reshape"(%150) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %565 = "tosa.mul"(%563, %564) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %566 = "tosa.reshape"(%149) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %567 = "tosa.add"(%565, %566) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %568 = "tosa.reluN"(%567) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %569 = "tosa.transpose"(%146, %276) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %570 = "tosa.conv2d"(%568, %569, %268) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %571 = "tosa.reshape"(%143) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %572 = "tosa.sub"(%570, %571) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %573 = "tosa.add"(%142, %277) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %574 = "tosa.rsqrt"(%573) : (tensor<64xf32>) -> tensor<64xf32>
    %575 = "tosa.reshape"(%574) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %576 = "tosa.mul"(%572, %575) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %577 = "tosa.reshape"(%145) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %578 = "tosa.mul"(%576, %577) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %579 = "tosa.reshape"(%144) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %580 = "tosa.add"(%578, %579) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %581 = "tosa.add"(%543, %580) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %582 = "tosa.transpose"(%141, %276) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %583 = "tosa.conv2d"(%581, %582, %269) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %584 = "tosa.reshape"(%138) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %585 = "tosa.sub"(%583, %584) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %586 = "tosa.add"(%137, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %587 = "tosa.rsqrt"(%586) : (tensor<384xf32>) -> tensor<384xf32>
    %588 = "tosa.reshape"(%587) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %589 = "tosa.mul"(%585, %588) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %590 = "tosa.reshape"(%140) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %591 = "tosa.mul"(%589, %590) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %592 = "tosa.reshape"(%139) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %593 = "tosa.add"(%591, %592) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %594 = "tosa.reluN"(%593) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %595 = "tosa.depthwise_conv2d"(%594, %136, %269) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %596 = "tosa.reshape"(%133) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %597 = "tosa.sub"(%595, %596) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %598 = "tosa.add"(%132, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %599 = "tosa.rsqrt"(%598) : (tensor<384xf32>) -> tensor<384xf32>
    %600 = "tosa.reshape"(%599) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %601 = "tosa.mul"(%597, %600) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %602 = "tosa.reshape"(%135) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %603 = "tosa.mul"(%601, %602) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %604 = "tosa.reshape"(%134) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %605 = "tosa.add"(%603, %604) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %606 = "tosa.reluN"(%605) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %607 = "tosa.transpose"(%131, %276) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %608 = "tosa.conv2d"(%606, %607, %268) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %609 = "tosa.reshape"(%128) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %610 = "tosa.sub"(%608, %609) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %611 = "tosa.add"(%127, %277) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %612 = "tosa.rsqrt"(%611) : (tensor<64xf32>) -> tensor<64xf32>
    %613 = "tosa.reshape"(%612) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %614 = "tosa.mul"(%610, %613) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %615 = "tosa.reshape"(%130) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %616 = "tosa.mul"(%614, %615) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %617 = "tosa.reshape"(%129) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %618 = "tosa.add"(%616, %617) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %619 = "tosa.add"(%581, %618) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %620 = "tosa.transpose"(%126, %276) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %621 = "tosa.conv2d"(%619, %620, %269) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %622 = "tosa.reshape"(%123) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %623 = "tosa.sub"(%621, %622) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %624 = "tosa.add"(%122, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %625 = "tosa.rsqrt"(%624) : (tensor<384xf32>) -> tensor<384xf32>
    %626 = "tosa.reshape"(%625) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %627 = "tosa.mul"(%623, %626) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %628 = "tosa.reshape"(%125) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %629 = "tosa.mul"(%627, %628) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %630 = "tosa.reshape"(%124) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %631 = "tosa.add"(%629, %630) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %632 = "tosa.reluN"(%631) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %633 = "tosa.depthwise_conv2d"(%632, %121, %269) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %634 = "tosa.reshape"(%118) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %635 = "tosa.sub"(%633, %634) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %636 = "tosa.add"(%117, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %637 = "tosa.rsqrt"(%636) : (tensor<384xf32>) -> tensor<384xf32>
    %638 = "tosa.reshape"(%637) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %639 = "tosa.mul"(%635, %638) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %640 = "tosa.reshape"(%120) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %641 = "tosa.mul"(%639, %640) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %642 = "tosa.reshape"(%119) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %643 = "tosa.add"(%641, %642) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %644 = "tosa.reluN"(%643) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %645 = "tosa.transpose"(%116, %276) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %646 = "tosa.conv2d"(%644, %645, %268) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %647 = "tosa.reshape"(%113) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %648 = "tosa.sub"(%646, %647) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %649 = "tosa.add"(%112, %277) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %650 = "tosa.rsqrt"(%649) : (tensor<64xf32>) -> tensor<64xf32>
    %651 = "tosa.reshape"(%650) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %652 = "tosa.mul"(%648, %651) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %653 = "tosa.reshape"(%115) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %654 = "tosa.mul"(%652, %653) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %655 = "tosa.reshape"(%114) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %656 = "tosa.add"(%654, %655) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %657 = "tosa.add"(%619, %656) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %658 = "tosa.transpose"(%111, %276) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %659 = "tosa.conv2d"(%657, %658, %269) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %660 = "tosa.reshape"(%108) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %661 = "tosa.sub"(%659, %660) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %662 = "tosa.add"(%107, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %663 = "tosa.rsqrt"(%662) : (tensor<384xf32>) -> tensor<384xf32>
    %664 = "tosa.reshape"(%663) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %665 = "tosa.mul"(%661, %664) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %666 = "tosa.reshape"(%110) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %667 = "tosa.mul"(%665, %666) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %668 = "tosa.reshape"(%109) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %669 = "tosa.add"(%667, %668) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %670 = "tosa.reluN"(%669) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %671 = "tosa.depthwise_conv2d"(%670, %106, %269) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %672 = "tosa.reshape"(%103) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %673 = "tosa.sub"(%671, %672) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %674 = "tosa.add"(%102, %277) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %675 = "tosa.rsqrt"(%674) : (tensor<384xf32>) -> tensor<384xf32>
    %676 = "tosa.reshape"(%675) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %677 = "tosa.mul"(%673, %676) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %678 = "tosa.reshape"(%105) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %679 = "tosa.mul"(%677, %678) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %680 = "tosa.reshape"(%104) {new_shape = [1, 1, 1, 384]} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %681 = "tosa.add"(%679, %680) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %682 = "tosa.reluN"(%681) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %683 = "tosa.transpose"(%101, %276) : (tensor<1x1x384x96xf32>, tensor<4xi32>) -> tensor<96x1x1x384xf32>
    %684 = "tosa.conv2d"(%682, %683, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x384xf32>, tensor<96x1x1x384xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %685 = "tosa.reshape"(%98) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %686 = "tosa.sub"(%684, %685) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %687 = "tosa.add"(%97, %277) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %688 = "tosa.rsqrt"(%687) : (tensor<96xf32>) -> tensor<96xf32>
    %689 = "tosa.reshape"(%688) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %690 = "tosa.mul"(%686, %689) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %691 = "tosa.reshape"(%100) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %692 = "tosa.mul"(%690, %691) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %693 = "tosa.reshape"(%99) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %694 = "tosa.add"(%692, %693) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %695 = "tosa.transpose"(%96, %276) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %696 = "tosa.conv2d"(%694, %695, %271) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %697 = "tosa.reshape"(%93) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %698 = "tosa.sub"(%696, %697) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %699 = "tosa.add"(%92, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %700 = "tosa.rsqrt"(%699) : (tensor<576xf32>) -> tensor<576xf32>
    %701 = "tosa.reshape"(%700) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %702 = "tosa.mul"(%698, %701) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %703 = "tosa.reshape"(%95) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %704 = "tosa.mul"(%702, %703) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %705 = "tosa.reshape"(%94) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %706 = "tosa.add"(%704, %705) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %707 = "tosa.reluN"(%706) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %708 = "tosa.depthwise_conv2d"(%707, %91, %271) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %709 = "tosa.reshape"(%88) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %710 = "tosa.sub"(%708, %709) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %711 = "tosa.add"(%87, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %712 = "tosa.rsqrt"(%711) : (tensor<576xf32>) -> tensor<576xf32>
    %713 = "tosa.reshape"(%712) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %714 = "tosa.mul"(%710, %713) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %715 = "tosa.reshape"(%90) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %716 = "tosa.mul"(%714, %715) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %717 = "tosa.reshape"(%89) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %718 = "tosa.add"(%716, %717) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %719 = "tosa.reluN"(%718) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %720 = "tosa.transpose"(%86, %276) : (tensor<1x1x576x96xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %721 = "tosa.conv2d"(%719, %720, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %722 = "tosa.reshape"(%83) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %723 = "tosa.sub"(%721, %722) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %724 = "tosa.add"(%82, %277) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %725 = "tosa.rsqrt"(%724) : (tensor<96xf32>) -> tensor<96xf32>
    %726 = "tosa.reshape"(%725) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %727 = "tosa.mul"(%723, %726) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %728 = "tosa.reshape"(%85) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %729 = "tosa.mul"(%727, %728) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %730 = "tosa.reshape"(%84) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %731 = "tosa.add"(%729, %730) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %732 = "tosa.add"(%694, %731) : (tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %733 = "tosa.transpose"(%81, %276) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %734 = "tosa.conv2d"(%732, %733, %271) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %735 = "tosa.reshape"(%78) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %736 = "tosa.sub"(%734, %735) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %737 = "tosa.add"(%77, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %738 = "tosa.rsqrt"(%737) : (tensor<576xf32>) -> tensor<576xf32>
    %739 = "tosa.reshape"(%738) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %740 = "tosa.mul"(%736, %739) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %741 = "tosa.reshape"(%80) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %742 = "tosa.mul"(%740, %741) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %743 = "tosa.reshape"(%79) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %744 = "tosa.add"(%742, %743) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %745 = "tosa.reluN"(%744) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %746 = "tosa.depthwise_conv2d"(%745, %76, %271) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %747 = "tosa.reshape"(%73) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %748 = "tosa.sub"(%746, %747) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %749 = "tosa.add"(%72, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %750 = "tosa.rsqrt"(%749) : (tensor<576xf32>) -> tensor<576xf32>
    %751 = "tosa.reshape"(%750) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %752 = "tosa.mul"(%748, %751) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %753 = "tosa.reshape"(%75) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %754 = "tosa.mul"(%752, %753) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %755 = "tosa.reshape"(%74) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %756 = "tosa.add"(%754, %755) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %757 = "tosa.reluN"(%756) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %758 = "tosa.transpose"(%71, %276) : (tensor<1x1x576x96xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %759 = "tosa.conv2d"(%757, %758, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %760 = "tosa.reshape"(%68) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %761 = "tosa.sub"(%759, %760) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %762 = "tosa.add"(%67, %277) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %763 = "tosa.rsqrt"(%762) : (tensor<96xf32>) -> tensor<96xf32>
    %764 = "tosa.reshape"(%763) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %765 = "tosa.mul"(%761, %764) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %766 = "tosa.reshape"(%70) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %767 = "tosa.mul"(%765, %766) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %768 = "tosa.reshape"(%69) {new_shape = [1, 1, 1, 96]} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %769 = "tosa.add"(%767, %768) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %770 = "tosa.add"(%732, %769) : (tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %771 = "tosa.transpose"(%66, %276) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %772 = "tosa.conv2d"(%770, %771, %271) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %773 = "tosa.reshape"(%63) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %774 = "tosa.sub"(%772, %773) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %775 = "tosa.add"(%62, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %776 = "tosa.rsqrt"(%775) : (tensor<576xf32>) -> tensor<576xf32>
    %777 = "tosa.reshape"(%776) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %778 = "tosa.mul"(%774, %777) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %779 = "tosa.reshape"(%65) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %780 = "tosa.mul"(%778, %779) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %781 = "tosa.reshape"(%64) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %782 = "tosa.add"(%780, %781) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %783 = "tosa.reluN"(%782) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %784 = "tosa.pad"(%783, %262) : (tensor<1x14x14x576xf32>, tensor<4x2xi32>) -> tensor<1x15x15x576xf32>
    %785 = "tosa.depthwise_conv2d"(%784, %61, %271) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x15x15x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %786 = "tosa.reshape"(%58) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %787 = "tosa.sub"(%785, %786) : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %788 = "tosa.add"(%57, %277) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %789 = "tosa.rsqrt"(%788) : (tensor<576xf32>) -> tensor<576xf32>
    %790 = "tosa.reshape"(%789) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %791 = "tosa.mul"(%787, %790) {shift = 0 : i32} : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %792 = "tosa.reshape"(%60) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %793 = "tosa.mul"(%791, %792) {shift = 0 : i32} : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %794 = "tosa.reshape"(%59) {new_shape = [1, 1, 1, 576]} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %795 = "tosa.add"(%793, %794) : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %796 = "tosa.reluN"(%795) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %797 = "tosa.transpose"(%56, %276) : (tensor<1x1x576x160xf32>, tensor<4xi32>) -> tensor<160x1x1x576xf32>
    %798 = "tosa.conv2d"(%796, %797, %272) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x576xf32>, tensor<160x1x1x576xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %799 = "tosa.reshape"(%53) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %800 = "tosa.sub"(%798, %799) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %801 = "tosa.add"(%52, %277) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %802 = "tosa.rsqrt"(%801) : (tensor<160xf32>) -> tensor<160xf32>
    %803 = "tosa.reshape"(%802) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %804 = "tosa.mul"(%800, %803) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %805 = "tosa.reshape"(%55) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %806 = "tosa.mul"(%804, %805) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %807 = "tosa.reshape"(%54) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %808 = "tosa.add"(%806, %807) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %809 = "tosa.transpose"(%51, %276) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %810 = "tosa.conv2d"(%808, %809, %273) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %811 = "tosa.reshape"(%48) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %812 = "tosa.sub"(%810, %811) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %813 = "tosa.add"(%47, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %814 = "tosa.rsqrt"(%813) : (tensor<960xf32>) -> tensor<960xf32>
    %815 = "tosa.reshape"(%814) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %816 = "tosa.mul"(%812, %815) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %817 = "tosa.reshape"(%50) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %818 = "tosa.mul"(%816, %817) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %819 = "tosa.reshape"(%49) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %820 = "tosa.add"(%818, %819) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %821 = "tosa.reluN"(%820) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %822 = "tosa.depthwise_conv2d"(%821, %46, %273) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %823 = "tosa.reshape"(%43) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %824 = "tosa.sub"(%822, %823) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %825 = "tosa.add"(%42, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %826 = "tosa.rsqrt"(%825) : (tensor<960xf32>) -> tensor<960xf32>
    %827 = "tosa.reshape"(%826) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %828 = "tosa.mul"(%824, %827) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %829 = "tosa.reshape"(%45) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %830 = "tosa.mul"(%828, %829) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %831 = "tosa.reshape"(%44) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %832 = "tosa.add"(%830, %831) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %833 = "tosa.reluN"(%832) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %834 = "tosa.transpose"(%41, %276) : (tensor<1x1x960x160xf32>, tensor<4xi32>) -> tensor<160x1x1x960xf32>
    %835 = "tosa.conv2d"(%833, %834, %272) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<160x1x1x960xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %836 = "tosa.reshape"(%38) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %837 = "tosa.sub"(%835, %836) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %838 = "tosa.add"(%37, %277) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %839 = "tosa.rsqrt"(%838) : (tensor<160xf32>) -> tensor<160xf32>
    %840 = "tosa.reshape"(%839) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %841 = "tosa.mul"(%837, %840) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %842 = "tosa.reshape"(%40) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %843 = "tosa.mul"(%841, %842) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %844 = "tosa.reshape"(%39) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %845 = "tosa.add"(%843, %844) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %846 = "tosa.add"(%808, %845) : (tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
    %847 = "tosa.transpose"(%36, %276) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %848 = "tosa.conv2d"(%846, %847, %273) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %849 = "tosa.reshape"(%33) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %850 = "tosa.sub"(%848, %849) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %851 = "tosa.add"(%32, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %852 = "tosa.rsqrt"(%851) : (tensor<960xf32>) -> tensor<960xf32>
    %853 = "tosa.reshape"(%852) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %854 = "tosa.mul"(%850, %853) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %855 = "tosa.reshape"(%35) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %856 = "tosa.mul"(%854, %855) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %857 = "tosa.reshape"(%34) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %858 = "tosa.add"(%856, %857) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %859 = "tosa.reluN"(%858) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %860 = "tosa.depthwise_conv2d"(%859, %31, %273) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %861 = "tosa.reshape"(%28) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %862 = "tosa.sub"(%860, %861) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %863 = "tosa.add"(%27, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %864 = "tosa.rsqrt"(%863) : (tensor<960xf32>) -> tensor<960xf32>
    %865 = "tosa.reshape"(%864) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %866 = "tosa.mul"(%862, %865) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %867 = "tosa.reshape"(%30) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %868 = "tosa.mul"(%866, %867) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %869 = "tosa.reshape"(%29) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %870 = "tosa.add"(%868, %869) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %871 = "tosa.reluN"(%870) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %872 = "tosa.transpose"(%26, %276) : (tensor<1x1x960x160xf32>, tensor<4xi32>) -> tensor<160x1x1x960xf32>
    %873 = "tosa.conv2d"(%871, %872, %272) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<160x1x1x960xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %874 = "tosa.reshape"(%23) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %875 = "tosa.sub"(%873, %874) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %876 = "tosa.add"(%22, %277) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %877 = "tosa.rsqrt"(%876) : (tensor<160xf32>) -> tensor<160xf32>
    %878 = "tosa.reshape"(%877) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %879 = "tosa.mul"(%875, %878) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %880 = "tosa.reshape"(%25) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %881 = "tosa.mul"(%879, %880) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %882 = "tosa.reshape"(%24) {new_shape = [1, 1, 1, 160]} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %883 = "tosa.add"(%881, %882) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %884 = "tosa.add"(%846, %883) : (tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
    %885 = "tosa.transpose"(%21, %276) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %886 = "tosa.conv2d"(%884, %885, %273) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %887 = "tosa.reshape"(%18) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %888 = "tosa.sub"(%886, %887) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %889 = "tosa.add"(%17, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %890 = "tosa.rsqrt"(%889) : (tensor<960xf32>) -> tensor<960xf32>
    %891 = "tosa.reshape"(%890) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %892 = "tosa.mul"(%888, %891) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %893 = "tosa.reshape"(%20) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %894 = "tosa.mul"(%892, %893) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %895 = "tosa.reshape"(%19) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %896 = "tosa.add"(%894, %895) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %897 = "tosa.reluN"(%896) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %898 = "tosa.depthwise_conv2d"(%897, %16, %273) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %899 = "tosa.reshape"(%13) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %900 = "tosa.sub"(%898, %899) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %901 = "tosa.add"(%12, %277) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %902 = "tosa.rsqrt"(%901) : (tensor<960xf32>) -> tensor<960xf32>
    %903 = "tosa.reshape"(%902) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %904 = "tosa.mul"(%900, %903) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %905 = "tosa.reshape"(%15) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %906 = "tosa.mul"(%904, %905) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %907 = "tosa.reshape"(%14) {new_shape = [1, 1, 1, 960]} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %908 = "tosa.add"(%906, %907) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %909 = "tosa.reluN"(%908) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %910 = "tosa.transpose"(%11, %276) : (tensor<1x1x960x320xf32>, tensor<4xi32>) -> tensor<320x1x1x960xf32>
    %911 = "tosa.conv2d"(%909, %910, %274) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x960xf32>, tensor<320x1x1x960xf32>, tensor<320xf32>) -> tensor<1x7x7x320xf32>
    %912 = "tosa.reshape"(%8) {new_shape = [1, 1, 1, 320]} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %913 = "tosa.sub"(%911, %912) : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %914 = "tosa.add"(%7, %277) : (tensor<320xf32>, tensor<1xf32>) -> tensor<320xf32>
    %915 = "tosa.rsqrt"(%914) : (tensor<320xf32>) -> tensor<320xf32>
    %916 = "tosa.reshape"(%915) {new_shape = [1, 1, 1, 320]} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %917 = "tosa.mul"(%913, %916) {shift = 0 : i32} : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %918 = "tosa.reshape"(%10) {new_shape = [1, 1, 1, 320]} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %919 = "tosa.mul"(%917, %918) {shift = 0 : i32} : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %920 = "tosa.reshape"(%9) {new_shape = [1, 1, 1, 320]} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %921 = "tosa.add"(%919, %920) : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %922 = "tosa.transpose"(%6, %276) : (tensor<1x1x320x1280xf32>, tensor<4xi32>) -> tensor<1280x1x1x320xf32>
    %923 = "tosa.conv2d"(%921, %922, %275) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x320xf32>, tensor<1280x1x1x320xf32>, tensor<1280xf32>) -> tensor<1x7x7x1280xf32>
    %924 = "tosa.reshape"(%3) {new_shape = [1, 1, 1, 1280]} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %925 = "tosa.sub"(%923, %924) : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %926 = "tosa.add"(%2, %277) : (tensor<1280xf32>, tensor<1xf32>) -> tensor<1280xf32>
    %927 = "tosa.rsqrt"(%926) : (tensor<1280xf32>) -> tensor<1280xf32>
    %928 = "tosa.reshape"(%927) {new_shape = [1, 1, 1, 1280]} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %929 = "tosa.mul"(%925, %928) {shift = 0 : i32} : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %930 = "tosa.reshape"(%5) {new_shape = [1, 1, 1, 1280]} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %931 = "tosa.mul"(%929, %930) {shift = 0 : i32} : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %932 = "tosa.reshape"(%4) {new_shape = [1, 1, 1, 1280]} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %933 = "tosa.add"(%931, %932) : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %934 = "tosa.reluN"(%933) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
    %935 = "tosa.reduce_sum"(%934) {axis = 1 : i64} : (tensor<1x7x7x1280xf32>) -> tensor<1x1x7x1280xf32>
    %936 = "tosa.reduce_sum"(%935) {axis = 2 : i64} : (tensor<1x1x7x1280xf32>) -> tensor<1x1x1x1280xf32>
    %937 = "tosa.reshape"(%936) {new_shape = [1, 1280]} : (tensor<1x1x1x1280xf32>) -> tensor<1x1280xf32>
    %938 = "tosa.reshape"(%278) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %939 = "tosa.mul"(%937, %938) {shift = 0 : i32} : (tensor<1x1280xf32>, tensor<1x1xf32>) -> tensor<1x1280xf32>
    %940 = "tosa.matmul"(%939, %1) : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
    %941 = "tosa.reshape"(%0) {new_shape = [1, 1000]} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %942 = "tosa.add"(%940, %941) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %943 = "tosa.exp"(%942) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %944 = "tosa.reduce_sum"(%943) {axis = 1 : i64} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
    %945 = "tosa.reciprocal"(%944) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %946 = "tosa.mul"(%943, %945) {shift = 0 : i32} : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    return %946 : tensor<1x1000xf32>
  }
}
