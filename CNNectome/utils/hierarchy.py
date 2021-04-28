from CNNectome.utils.label import Label

hierarchy = dict()
hierarchy['ecs'] = Label('ecs', 1)
hierarchy['plasma_membrane'] = Label('plasma_membrane', 2)
hierarchy['mito'] = Label('mito', (3, 4, 5))
hierarchy['mito_membrane'] = Label('mito_membrane', 3, scale_loss=False, scale_key=hierarchy['mito'].scale_key)
hierarchy['mito_DNA'] = Label('mito_DNA', 5, scale_loss=False, scale_key=hierarchy['mito'].scale_key)
hierarchy['golgi'] = Label('golgi', (6, 7))
hierarchy['golgi_membrane'] = Label('golgi_membrane', 6)
hierarchy['vesicle'] = Label('vesicle', (8, 9))
hierarchy['vesicle_membrane'] = Label('vesicle_membrane', 8, scale_loss=False, scale_key=hierarchy['vesicle'].scale_key)
hierarchy['MVB'] = Label('MVB', (10, 11), )
hierarchy['MVB_membrane'] = Label('MVB_membrane', 10, scale_loss=False, scale_key=hierarchy['MVB'].scale_key)
hierarchy['lysosome'] = Label('lysosome', (12, 13))
hierarchy['lysosome_membrane'] = Label('lysosome_membrane', 12, scale_loss=False,
                                       scale_key=hierarchy['lysosome'].scale_key)
hierarchy['LD'] = Label('LD', (14, 15))
hierarchy['LD_membrane'] = Label('LD_membrane', 14, scale_loss=False, scale_key=hierarchy['LD'].scale_key)
hierarchy['er'] = Label('er', (16, 17, 18, 19, 20, 21, 22, 23))
hierarchy['er_membrane'] = Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=hierarchy['er'].scale_key)
hierarchy['ERES'] = Label('ERES', (18, 19))
hierarchy['ERES_membrane'] = Label('ERES_membrane', 18, scale_loss=False, scale_key=hierarchy['ERES'].scale_key)
hierarchy['nucleus'] = Label('nucleus', (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37), generic_label=37)
hierarchy['nucleolus'] = Label('nucleolus', 29, separate_labelset=True)
hierarchy['NE'] = Label('NE', (20, 21, 22, 23))
hierarchy['NE_membrane'] = Label('NE_membrane', (20, 22, 23), scale_loss=False, scale_key=hierarchy['NE'].scale_key)
hierarchy['nuclear_pore'] = Label('nuclear_pore', (22, 23))
hierarchy['nuclear_pore_out'] = Label('nuclear_pore_out', 22, scale_loss=False,
                                      scale_key=hierarchy['nuclear_pore'].scale_key)
hierarchy['chromatin'] = Label('chromatin', (24, 25, 26, 27))
hierarchy['NHChrom'] = Label('NHChrom', 25)
hierarchy['EChrom'] = Label('EChrom', 26)
hierarchy['NEChrom'] = Label('NEChrom', 27)
hierarchy['microtubules'] = Label('microtubules', (30, 36))
hierarchy['microtubules_out'] = Label('microtubules_out', 30, scale_loss=False,
                                      scale_key=hierarchy['microtubules'].scale_key)
hierarchy['centrosome'] = Label('centrosome', 31, add_constant=2, separate_labelset=True)
hierarchy['distal_app'] = Label('distal_app', 32)
hierarchy['subdistal_app'] = Label('subdistal_app', 33)
hierarchy['ribosomes'] = Label('ribosomes', 34, add_constant=8, separate_labelset=True)
hierarchy['cytosol'] = Label('cytosol', 35)

short_names = dict()
short_names["actin"] = "Actin"
short_names["centrosome"] = "Centrosome"
short_names["distal_app"] = "Centrosome D App"
short_names["subdistal_app"] = "Centrosome SD App"
short_names["chromatin"] = "Chromatin"
short_names["cytosol"] = "Cytosol"
short_names["er"] = "ER"
short_names["ERES"] = "ERES"
short_names["ERES_lumen"] = "ERES lum"
short_names["ERES_membrane"] = "ERES mem"
short_names["er_lumen"] = "ER lum"
short_names["er_membrane"] = "ER mem"
short_names["MVB"] = "Endo"
short_names["MVB_lumen"] = "Endo lum"
short_names["MVB_membrane"] = "Endo mem"
short_names["EChrom"] = "E Chrom"
short_names["ecs"] = "ECS"
short_names["golgi"] = "Golgi"
short_names["golgi_lumen"] = "Golgi lum"
short_names["golgi_membrane"] = "Golgi mem"
short_names["HChrom"] = "H Chrom"
short_names["LD"] = "LD"
short_names["LD_lumen"] = "LD lum"
short_names["LD_membrane"] = "LD mem"
short_names["lysosome"] = "Lyso"
short_names["lysosome_lumen"] = "Lyso lum"
short_names["lysosome_membrane"] = "Lyso mem"
short_names["microtubules"] = "MT"
short_names["microtubules_in"] = "MT in"
short_names["microtubules_out"] = "MT out"
short_names["mito"] = "Mito"
short_names["mito_lumen"] = "Mito lum"
short_names["mito_membrane"] = "Mito mem"
short_names["mito_DNA"] = "Mito Ribo"
short_names["NE"] = "NE"
short_names["NE_lumen"] = "NE lum"
short_names["NE_membrane"] = "NE mem"
short_names["nuclear_pore"] = "NP"
short_names["nuclear_pore_in"] = "NP in"
short_names["nuclear_pore_out"] = "NP out"
short_names["nucleoplasm"] = "Nucleoplasm"
short_names["nucleolus"] = "Nucleolus"
short_names["NEChrom"] = "N-E Chrom"
short_names["NHChrom"] = "N-H Chrom"
short_names["nucleus"] = "Nucleus"
short_names["plasma_membrane"] = "PM"
short_names["ribosomes"] = "Ribo"
short_names["vesicle"] = "Vesicle"
short_names["vesicle_lumen"] = "Vesicle lum"
short_names["vesicle_membrane"] = "Vesicle mem"

long_names = dict()
long_names["actin"] = "Actin"
long_names["centrosome"] = "Centrosome"
long_names["distal_app"] = "Centrosome Distal Appendage"
long_names["subdistal_app"] = "Centrosome Subdistal Appendage"
long_names["chromatin"] = "Chromatin"
long_names["cytosol"] = "Cytosol"
long_names["er"] = "Endoplasmic Reticulum"
long_names["ERES"] = "Endoplasmic Reticulum Exit Site"
long_names["ERES_lumen"] = "Endoplasmic Reticulum Exit Site lumen"
long_names["ERES_membrane"] = "Endoplasmic Reticulum Exit Site membrane"
long_names["er_lumen"] = "Endoplasmic Reticulum lumen"
long_names["er_membrane"] = "Endoplasmic Reticulum membrane"
long_names["MVB"] = "Endosomal Network"
long_names["MVB_lumen"] = "Endosome lumen"
long_names["MVB_membrane"] = "Endosome membrane"
long_names["EChrom"] = "Euchromatin"
long_names["ecs"] = "Extracellular Space"
long_names["golgi"] = "Golgi"
long_names["golgi_lumen"] = "Golgi lumen"
long_names["golgi_membrane"] = "Golgi membrane"
long_names["HChrom"] = "Heterochromatin"
long_names["LD"] = "Lipid Droplet"
long_names["LD_lumen"] = "Lipid Droplet lumen"
long_names["LD_membrane"] = "Lipid Droplet membrane"
long_names["lysosome"] = "Lysosome"
long_names["lysosome_lumen"] = "Lysosome lumen"
long_names["lysosome_membrane"] = "Lysosome membrane"
long_names["microtubules"] = "Microtubule"
long_names["microtubules_in"] = "Microtubule inner"
long_names["microtubules_out"] = "Microtubule outer"
long_names["mito"] = "Mitochondria"
long_names["mito_lumen"] = "Mitochondria lumen"
long_names["mito_membrane"] = "Mitochondria membrane"
long_names["mito_DNA"] = "Mitochondria Ribosome"
long_names["NE"] = "Nuclear Envelope"
long_names["NE_lumen"] = "Nuclear Envelope lumen"
long_names["NE_membrane"] = "Nuclear Envelope membrane"
long_names["nuclear_pore"] = "Nuclear Pore"
long_names["nuclear_pore_in"] = "Nuclear Pore inner"
long_names["nuclear_pore_out"] = "Nuclear Pore outer"
long_names["nucleoplasm"] = "Nucleoplasm"
long_names["nucleolus"] = "Nucleolus"
long_names["NEChrom"] = "Nucleolus associated Euchromatin"
long_names["NHChrom"] = "Nucleolus associated Heterochromatin"
long_names["nucleus"] = "Nucleus"
long_names["plasma_membrane"] = "Plasma Membrane"
long_names["ribosomes"] = "Ribosome"
long_names["vesicle"] = "Vesicle"
long_names["vesicle_lumen"] = "Vesicle lumen"
long_names["vesicle_membrane"] = "Vesicle membrane"
