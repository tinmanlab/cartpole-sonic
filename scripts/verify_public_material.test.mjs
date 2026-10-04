import test from 'node:test';
import assert from 'node:assert/strict';
import {contextualReferences,machineLocations,auditPublicMaterial} from './verify_public_material.mjs';

test('public material rejects unrelated first-party repository references without deleting public upstream credits',()=>{
  const options={owner:'example-owner',allowed:new Set(['public-lab'])};
  assert.deepEqual(contextualReferences('https://github.com/example-owner/public-lab.git',options),[]);
  assert.deepEqual(contextualReferences('https://github.com/example-owner/another-project',options),['another-project']);
  assert.deepEqual(contextualReferences('https://github.com/NVlabs/GR00T-WholeBodyControl',options),[]);
});
test('personal machine locations are not needed for generic prose',()=>{
  assert.equal(machineLocations('/home/example-person/project/'),true);
  assert.equal(machineLocations('C:\\Users\\example-person\\project'),true);
  assert.equal(machineLocations('/path/to/pinned/upstream'),false);
});
test('current public files have no unreviewed project coupling or broken document links',()=>{
  assert.deepEqual(auditPublicMaterial().problems,[]);
});
