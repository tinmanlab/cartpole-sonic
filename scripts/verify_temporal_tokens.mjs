import {TemporalTokenLab} from "../temporal_token_lab.js";

const lab=new TemporalTokenLab();
const before=lab.snapshot().baseline;
lab.train(200);
const after=lab.snapshot().current;

if(!(after.mseTwo <= after.mseOne*.65)){
  throw new Error("2-token reconstruction did not beat 1-token baseline by required margin");
}
if(!(after.uniqueCodesTwo >= 60)){
  throw new Error("2-token model did not use enough distinct FSQ combinations");
}
if(!(after.uniqueCodesTwo > after.uniqueCodesOne*2)){
  throw new Error("2-token code usage did not materially exceed 1-token usage");
}

console.log(JSON.stringify({
  schema:"temporal-token-capacity/v1",
  before,
  after,
  interpretation:"Capacity ablation only: more token slots increase discrete representational capacity in this toy. It does not prove that two tokens are universally optimal.",
},null,2));
