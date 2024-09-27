import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from 'langchain/hub';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { RunnableSequence,RunnablePassthrough } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { SystemMessage } from "@langchain/core/messages";
import { createRetrieverTool } from "langchain/tools/retriever";
import { z } from "zod";

export class RagWebClient {
    
    private vectorStore?: MemoryVectorStore;
    private prompt?: ChatPromptTemplate<any,any>;

    constructor(private url:string) {
        
    }
    
    public async getChain() {
        if (!this.vectorStore){ 
            this.vectorStore = await this.initVectorStore();
        }

        this.prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
        //this.prompt.promptMessages.push(new SystemMessage('Always response to the user in spanish'));

        const llm = new ChatOpenAI({model: 'gpt-4o-mini',temperature: 0});
        // const chain = await createStuffDocumentsChain(
        //     {
        //         llm: model,
        //         prompt: this.prompt,
        //         outputParser: new StringOutputParser()
        //     }
        // );

        if(this.vectorStore){
            const retriever = this.vectorStore.asRetriever();
            let chain2 = RunnableSequence.from([
                    { 
                    context: retriever.pipe(formatDocumentsAsString), 
                    question: new RunnablePassthrough()
                },
                this.prompt,
                llm,
                new StringOutputParser()
            ]);
            
            return chain2;
        }
        return null;
        
    }

    public async getRetrieverAsTool() {
        if (!this.vectorStore) {
            this.vectorStore = await this.initVectorStore();
        }

        const tool = this.vectorStore.asRetriever().asTool({name: "uach",
                                                            description: "Informacion al 2024 relativa a la carrera de ingenieria aeroespacial impartida por la UACH (universidad autonoma de chiahuahua)",
                                                            schema: z.string()
                                                        });

        return tool;
    }

    private async initVectorStore():Promise<MemoryVectorStore> {
        const loader = new CheerioWebBaseLoader(this.url);
        const docs =await  loader.load();

        const text_splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });

        const splits = await text_splitter.splitDocuments(docs);
        const vector_store = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());

        return vector_store;
    }




    

}

