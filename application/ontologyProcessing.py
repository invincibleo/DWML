import json


class OntologyProcessing(object):

    @staticmethod
    def read_ontology(filename):
        # 0. read AudioSet Ontology data
        with open(filename) as data_file:
            raw_aso = json.load(data_file)
        return raw_aso

    @staticmethod
    def form_dictionary(raw_aso):
        # 1. format data as a dictionary
        ## aso["/m/0dgw9r"] > {'restrictions': [u'abstract'], 'child_ids': [u'/m/09l8g', u'/m/01w250', u'/m/09hlz4', u'/m/0bpl036', u'/m/0160x5', u'/m/0k65p', u'/m/01jg02', u'/m/04xp5v', u'/t/dd00012'], 'name': u'Human sounds'}
        aso = {}
        for category in raw_aso:
            tmp = dict()
            tmp["name"] = category["name"]
            tmp["restrictions"] = category["restrictions"]
            tmp["child_ids"] = category["child_ids"]
            tmp["parents_ids"] = []
            aso[category["id"]] = tmp
        return aso

    @staticmethod
    def fetch_parents(aso):
        # 2. fetch higher_categories > ["/m/0dgw9r","/m/0jbk","/m/04rlf","/t/dd00098","/t/dd00041","/m/059j3w","/t/dd00123"]
        for cat in aso: # find parents
            for c in aso[cat]["child_ids"]:
                aso[c]["parents_ids"].append(cat)
        return aso

    @staticmethod
    def get_label_name_list(filename):
        raw_aso = OntologyProcessing.read_ontology(filename)
        aso = OntologyProcessing.form_dictionary(raw_aso)
        aso = OntologyProcessing.fetch_parents(aso)
        return aso

